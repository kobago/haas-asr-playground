"""Benchmark one ASR model on asr-ja-bench subsets: CER, speed, memory.

python bench.py --model haas|whisper --bench DIR --subset fleurs [--dtype bf16|fp16|fp32]
                [--batch-size N] [--out results.jsonl]

Each run is its own process so that memory numbers are not mixed between models.
GPU memory is reported two ways: torch's peak allocated/reserved, and the peak
process memory seen by nvidia-smi (includes CUDA context, allocator caching).
"""

import argparse
import gc
import json
import os
import resource
import subprocess
import threading
import time
from pathlib import Path

import jacer
import numpy as np
import soundfile as sf
import torch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class GpuMonitor(threading.Thread):
    """Poll nvidia-smi for this process's GPU memory (MiB) and keep the peak."""

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval, self.peak, self.stop_flag, self.pid = interval, 0, False, os.getpid()

    def run(self):
        while not self.stop_flag:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            for line in out.splitlines():
                pid, mem = (x.strip() for x in line.split(","))
                if int(pid) == self.pid:
                    self.peak = max(self.peak, int(mem))
            time.sleep(self.interval)


def load_rows(bench, subset, limit):
    rows = [json.loads(x) for x in open(bench / f"{subset}.jsonl", encoding="utf-8") if x.strip()]
    rows = rows[:limit] if limit else rows
    for r in rows:
        r["wav"] = sf.read(bench / r["audio"], dtype="float32")[0]
    return rows


def make_haas(dtype, batch_size, cuda_graph=True):
    import haas_asr
    from omegaconf import open_dict

    model = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", dtype=dtype)
    if not cuda_graph:
        cfg = model.cfg.decoding
        with open_dict(cfg):
            cfg.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(cfg, verbose=False)

    def run(wavs):
        out = []
        for i in range(0, len(wavs), batch_size):
            b = wavs[i : i + batch_size]
            out += [h.text.strip() for h in model.transcribe(b, batch_size=len(b), verbose=False)]
        return out

    return model, run


def make_whisper(dtype, batch_size, model_id="openai/whisper-large-v3"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa").to("cuda").eval()

    def run(wavs):
        out = []
        # Sort by length so padding of the decoder is small; restore order after.
        order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
        res = {}
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            feats = proc.feature_extractor(
                [wavs[j] for j in idx], sampling_rate=16000, return_tensors="pt")
            feats = feats.input_features.to("cuda", dtype=dtype)
            with torch.inference_mode():
                ids = model.generate(feats, language="ja", task="transcribe", num_beams=1,
                                     do_sample=False, max_new_tokens=440)
            for j, t in zip(idx, proc.batch_decode(ids, skip_special_tokens=True), strict=True):
                res[j] = t.strip()
        out = [res[i] for i in range(len(wavs))]
        return out

    return model, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--subset", action="append", required=True)
    ap.add_argument("--dtype", default="bf16", choices=DTYPES)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--hyp-dir", type=Path)
    ap.add_argument("--no-cuda-graph", action="store_true")
    a = ap.parse_args()
    dtype = DTYPES[a.dtype]

    mon = GpuMonitor()
    mon.start()
    t0 = time.perf_counter()
    if a.model == "haas":
        model, run = make_haas(dtype, a.batch_size, not a.no_cuda_graph)
    elif a.model.startswith("whisper"):
        mid = {"whisper": "openai/whisper-large-v3",
               "whisper-turbo": "openai/whisper-large-v3-turbo"}[a.model]
        model, run = make_whisper(dtype, a.batch_size, mid)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    # NeMo's restore + dtype cast leaves the float32 copies in reference cycles.
    load_peak_mib = torch.cuda.max_memory_allocated() / 2**20
    gc.collect()
    torch.cuda.empty_cache()
    params = sum(p.numel() for p in model.parameters())
    weights_mib = torch.cuda.memory_allocated() / 2**20

    # Warm-up (CUDA kernels, Triton JIT) on a few samples, not timed.
    warm = load_rows(a.bench, a.subset[0], 8)
    run([r["wav"] for r in warm])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for subset in a.subset:
        rows = load_rows(a.bench, subset, a.limit)
        audio_s = sum(len(r["wav"]) for r in rows) / 16000
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        hyps = run([r["wav"] for r in rows])
        torch.cuda.synchronize()
        el = time.perf_counter() - t
        cer = jacer.cer([r["text"] for r in rows], hyps, jacer.METRICS)
        rec = {
            "model": a.model, "dtype": a.dtype, "cuda_graph": not a.no_cuda_graph, "batch_size": a.batch_size, "subset": subset,
            "samples": len(rows), "audio_s": round(audio_s, 1), "elapsed_s": round(el, 2),
            "rtfx": round(audio_s / el, 1), "load_s": round(load_s, 1), "load_peak_MiB": round(load_peak_mib),
            "params_M": round(params / 1e6, 1), "weights_MiB": round(weights_mib),
            "peak_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20),
            "peak_nvsmi_MiB": mon.peak,
            "peak_rss_MiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            **{k: round(v * 100, 2) for k, v in cer.items()},
        }
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if a.out:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if a.hyp_dir:
            a.hyp_dir.mkdir(parents=True, exist_ok=True)
            with open(a.hyp_dir / f"{a.model}_{a.dtype}_{subset}.jsonl", "w") as f:
                for r, h in zip(rows, hyps, strict=True):
                    f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h},
                                       ensure_ascii=False) + "\n")
    time.sleep(0.3)
    mon.stop_flag = True


if __name__ == "__main__":
    main()
