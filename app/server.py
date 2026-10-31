"""Haas-ja-v1.0 playground: a single-page app and an API server.

    ./start.sh    (see README.md)

Endpoints (`model` is one of the ids in /api/info; default: the first)
    GET  /                          the SPA (static/index.html)
    GET  /api/info                  available models
    POST /api/transcribe?model=ID   audio file in the body (any format ffmpeg reads) -> transcript
    WS   /ws/realtime?model=ID      live captions from microphone frames (protocol: demos/realtime.py)

Models
    haas-gpu    the original model, PyTorch + NeMo on the GPU
    onnx-q      oboroge0/Haas-ja-v1.0-onnx-quantized, sherpa-onnx on the CPU
    onnx-fp32   oboroge0/Haas-ja-v1.0-onnx, sherpa-onnx on the CPU

The GPU model is tuned for an 8 GB Turing GPU (e.g. RTX 2060 SUPER): float16 instead of
bfloat16 (no native bf16 on Turing, ~3.6x slower), NeMo's CUDA-graph decoder off (illegal
memory access at batch >= 32 on this setup), and batches capped at 16. The ONNX models use
6 threads, the number of performance cores on an i5-13400F (more threads land on the
efficiency cores and are slower).
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import gc
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download

HAAS_REPO = Path(os.environ.get("HAAS_REPO", Path.home() / "projects/github.com/Haas-ASR"))
sys.path.insert(0, str(HAAS_REPO))  # for the demos package (realtime WebSocket, segmenter)

import haas_asr  # noqa: E402
from haas_asr.audio import SAMPLE_RATE  # noqa: E402
from omegaconf import open_dict  # noqa: E402

from demos import realtime  # noqa: E402

STATIC = Path(__file__).parent / "static"
MAX_BATCH = 16
MAX_UPLOAD_BYTES = 300 * 1_000_000
MODEL_CACHE = Path.home() / ".cache/haas-mic-app/models"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
ONNX_MODELS = {
    "onnx-q": ("oboroge0/Haas-ja-v1.0-onnx-quantized", "encoder.quantized.onnx",
               "ONNX 量子化版"),
    "onnx-fp32": ("oboroge0/Haas-ja-v1.0-onnx", "encoder.onnx", "ONNX fp32版"),
}


class Backend:
    """One model on its own worker thread; same interface as demos.inference.Transcriber."""

    id: str
    label: str
    device_name: str
    precision: str

    def __init__(self) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix=self.id)

    def _transcribe(self, audios: list[np.ndarray]) -> list[str]:
        raise NotImplementedError

    def warm_up(self) -> None:
        noise = np.random.default_rng(0).normal(0, 0.01, 30 * SAMPLE_RATE).astype(np.float32)
        for batch in self._warm_up_batches(noise):
            self._executor.submit(self._transcribe, batch).result()

    def _warm_up_batches(self, noise: np.ndarray) -> list[list[np.ndarray]]:
        return [[noise[:SAMPLE_RATE]]]

    async def transcribe(
        self, audios: list[np.ndarray], batch_size: int = 1
    ) -> tuple[list[str], float]:
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        texts: list[str] = []
        for i in range(0, len(audios), MAX_BATCH):
            texts += await loop.run_in_executor(
                self._executor, self._transcribe, audios[i : i + MAX_BATCH]
            )
        return texts, time.perf_counter() - started

    def info(self) -> dict:
        return {"id": self.id, "label": self.label, "device": self.device_name,
                "precision": self.precision}


class NemoBackend(Backend):
    id = "haas-gpu"
    label = "オリジナル (GPU)"

    def __init__(self, model: str, dtype: str):
        super().__init__()
        self.precision = dtype
        self._executor.submit(self._load, model, DTYPES[dtype]).result()
        self.device_name = torch.cuda.get_device_name()

    def _load(self, model: str, dtype: torch.dtype) -> None:
        self._model = haas_asr.load_model(model, dtype=dtype)
        cfg = self._model.cfg.decoding
        with open_dict(cfg):
            cfg.greedy.use_cuda_graph_decoder = False
        self._model.change_decoding_strategy(cfg, verbose=False)
        # The dtype cast leaves the float32 weights in reference cycles (~1.2 GB).
        gc.collect()
        torch.cuda.empty_cache()

    def _warm_up_batches(self, noise: np.ndarray) -> list[list[np.ndarray]]:
        return [[noise[:SAMPLE_RATE]], [noise] * MAX_BATCH]

    def _transcribe(self, audios: list[np.ndarray]) -> list[str]:
        with torch.inference_mode():
            out = self._model.transcribe(audios, batch_size=len(audios), verbose=False)
        return [h.text.strip() for h in out]


class OnnxBackend(Backend):
    def __init__(self, model_id: str, threads: int):
        import sherpa_onnx

        self.id = model_id
        repo, encoder, self.label = ONNX_MODELS[model_id]
        super().__init__()
        # Real files in one folder, not the hub cache: there encoder.onnx and encoder.data
        # are symlinks into different blob folders, and onnxruntime refuses external data
        # that resolves outside the model's folder.
        d = Path(snapshot_download(
            repo, allow_patterns=["*.onnx", "*.data", "tokens.txt"],
            local_dir=MODEL_CACHE / model_id,
        ))
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / encoder), decoder=str(d / "decoder.onnx"),
            joiner=str(d / "joiner.onnx"), tokens=str(d / "tokens.txt"),
            num_threads=threads, model_type="nemo_transducer", feature_dim=128,
        )
        self.device_name = f"CPU {threads} スレッド"
        self.precision = "int8/int4" if model_id == "onnx-q" else "fp32"

    def _transcribe(self, audios: list[np.ndarray]) -> list[str]:
        texts = []
        for audio in audios:  # one at a time: decode_streams pads to the longest
            stream = self._rec.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            self._rec.decode_stream(stream)
            texts.append(stream.result.text.strip())
        return texts


def decode(raw: bytes) -> np.ndarray:
    """Any container/codec ffmpeg reads (WAV, MP3, M4A, WebM, ...) -> 16 kHz mono float32."""
    # A file, not a pipe: MP4/M4A often keep their index at the end and need seeking.
    with tempfile.NamedTemporaryFile(prefix="haas-upload-") as f:
        f.write(raw)
        f.flush()
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", f.name, "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-f", "f32le", "pipe:1"],
            capture_output=True,
        )
    if proc.returncode != 0:
        raise HTTPException(415, f"音声を読み込めませんでした: {proc.stderr.decode()[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def create_app(backends: dict[str, Backend], model_name: str) -> FastAPI:
    app = FastAPI(title="Haas playground")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    default = next(iter(backends))

    def pick(model: str | None) -> Backend:
        backend = backends.get(model or default)
        if backend is None:
            raise HTTPException(404, f"モデル {model} はありません: {', '.join(backends)}")
        return backend

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info() -> dict:
        return {"model": model_name, "models": [b.info() for b in backends.values()]}

    @app.post("/api/transcribe")
    async def transcribe(request: Request, model: str | None = None) -> dict:
        backend = pick(model)
        raw = await request.body()
        if not 0 < len(raw) <= MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{MAX_UPLOAD_BYTES // 1_000_000} MB までの音声を送ってください")
        audio = await asyncio.to_thread(decode, raw)
        duration = audio.size / SAMPLE_RATE
        if duration < 0.1:
            raise HTTPException(400, "音声が短すぎます")
        bounds = haas_asr.split_at_quiet_points(audio)
        texts, seconds = await backend.transcribe([audio[s:e] for s, e in bounds])
        segments = [
            {"start": s / SAMPLE_RATE, "end": e / SAMPLE_RATE, "text": t}
            for (s, e), t in zip(bounds, texts, strict=True)
            if t
        ]
        return {
            "model": backend.id,
            "text": "".join(seg["text"] for seg in segments),
            "segments": segments,
            "duration": duration,
            "elapsed": seconds,
            "rtfx": duration / seconds if seconds else None,
        }

    @app.websocket("/ws/realtime")
    async def realtime_socket(websocket: WebSocket, model: str | None = None) -> None:
        backend = backends.get(model or default)
        if backend is None:
            await websocket.close(code=4404)
            return
        await realtime.serve(websocket, backend)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="third-intelligence/Haas-ja-v1.0")
    p.add_argument("--dtype", default="fp16", choices=DTYPES)
    p.add_argument("--onnx", nargs="*", default=list(ONNX_MODELS), choices=list(ONNX_MODELS),
                   help="ONNX models to load as well (default: all; none: --onnx)")
    p.add_argument("--onnx-threads", type=int, default=6)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--ssl-certfile")
    p.add_argument("--ssl-keyfile")
    a = p.parse_args()

    torch.set_num_threads(min(16, os.cpu_count() or 1))
    backends: dict[str, Backend] = {}
    print(f"Loading {a.model} ({a.dtype}) ...", flush=True)
    backends["haas-gpu"] = NemoBackend(a.model, a.dtype)
    for model_id in a.onnx:
        print(f"Loading {ONNX_MODELS[model_id][0]} ...", flush=True)
        backends[model_id] = OnnxBackend(model_id, a.onnx_threads)
    for backend in backends.values():
        backend.warm_up()
    scheme = "https" if a.ssl_certfile else "http"
    print(f"Ready ({', '.join(backends)}): {scheme}://{a.host}:{a.port}", flush=True)
    uvicorn.run(
        create_app(backends, Path(a.model).name), host=a.host, port=a.port,
        ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile,
        http="h11", ws="wsproto", log_level="warning",
    )


if __name__ == "__main__":
    main()
