"""Benchmark the sherpa-onnx conversions of Haas-ja-v1.0 on asr-ja-bench subsets (CPU).

python bench_onnx.py --model-dir DIR --encoder encoder.onnx --bench DIR --subset fleurs
                     [--threads 4] [--batch 1] [--provider cpu] [--out results.jsonl]

Scored with jacer like bench.py, so numbers are comparable with the PyTorch runs.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import jacer
import sherpa_onnx
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--subset", action="append", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1, help="streams per decode_streams call")
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--hyp-dir", type=Path)
    a = ap.parse_args()

    d = a.model_dir
    t0 = time.perf_counter()
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / a.encoder), decoder=str(d / "decoder.onnx"),
        joiner=str(d / "joiner.onnx"), tokens=str(d / "tokens.txt"),
        num_threads=a.threads, model_type="nemo_transducer", feature_dim=128,
        provider=a.provider,
    )
    load_s = time.perf_counter() - t0
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def run(paths):
        """Read audio lazily (so RSS is the model's, not the dataset's); time decoding only."""
        out, audio_s, busy = [], 0.0, 0.0
        for i in range(0, len(paths), a.batch):
            batch = [sf.read(p, dtype="float32")[0] for p in paths[i : i + a.batch]]
            audio_s += sum(len(w) for w in batch) / 16000
            t = time.perf_counter()  # features (accept_waveform) + decoding
            streams = []
            for w in batch:
                s = rec.create_stream()
                s.accept_waveform(16000, w)
                streams.append(s)
            if len(streams) == 1:
                rec.decode_stream(streams[0])
            else:
                rec.decode_streams(streams)
            busy += time.perf_counter() - t
            out += [s.result.text.strip() for s in streams]
        return out, audio_s, busy

    # Decode every subset before scoring: jacer loads a dictionary that would
    # otherwise be counted in the peak RSS.
    decoded = []
    for subset in a.subset:
        rows = [json.loads(x) for x in open(a.bench / f"{subset}.jsonl", encoding="utf-8")]
        rows = rows[: a.limit] if a.limit else rows
        paths = [a.bench / r["audio"] for r in rows]
        if subset == a.subset[0]:
            run(paths[:4])  # warm-up
        hyps, audio_s, el = run(paths)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        decoded.append((subset, rows, hyps, audio_s, el, peak))
        print(f"{subset}: decoded in {el:.1f}s", flush=True)

    for subset, rows, hyps, audio_s, el, peak in decoded:
        cer = jacer.cer([r["text"] for r in rows], hyps, jacer.METRICS)
        res = {
            "model": a.label, "provider": a.provider, "threads": a.threads, "batch": a.batch,
            "subset": subset, "samples": len(rows), "audio_s": round(audio_s, 1),
            "elapsed_s": round(el, 2), "rtfx": round(audio_s / el, 1),
            "rtf": round(el / audio_s, 4), "load_s": round(load_s, 1),
            "rss_after_load_MiB": round(rss_after_load), "peak_rss_MiB": round(peak),
            **{k: round(v * 100, 2) for k, v in cer.items()},
        }
        print(json.dumps(res, ensure_ascii=False), flush=True)
        if a.out:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        if a.hyp_dir:
            a.hyp_dir.mkdir(parents=True, exist_ok=True)
            with open(a.hyp_dir / f"{a.label}_t{a.threads}_{subset}.jsonl", "w") as f:
                for r, h in zip(rows, hyps, strict=True):
                    f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h},
                                       ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
