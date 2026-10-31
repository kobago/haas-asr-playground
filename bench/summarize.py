"""Print the result tables of REPORT.md from the result files.

python summarize.py ../results

Overall CER is weighted by audio length, as the asr-ja-bench leaderboard does;
overall RTFx is total audio / total decoding time.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ["cer_strict", "cer_flex", "cer_kana"]


def table(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    print(f"| {'run':<34} | {'subset':<7} | strict | flex | kana | RTFx |")
    print("|---|---|---|---|---|---|")
    for name, rs in groups.items():
        for r in rs:
            print(f"| {name:<34} | {r['subset']:<7} | " + " | ".join(f"{r[m]:.2f}" for m in METRICS)
                  + f" | {r['rtfx']:.1f} |")
        audio = sum(r["audio_s"] for r in rs)
        cer = [sum(r[m] * r["audio_s"] for r in rs) / audio for m in METRICS]
        rtfx = audio / sum(r["elapsed_s"] for r in rs)
        print(f"| {name:<34} | {'overall':<7} | " + " | ".join(f"{c:.2f}" for c in cer)
              + f" | {rtfx:.1f} |")
    print()


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    load = lambda name: [json.loads(x) for x in open(d / name, encoding="utf-8") if x.strip()]
    print("## GPU (results.jsonl)")
    table(load("results.jsonl"), lambda r: f"{r['model']} {r['dtype']} bs{r['batch_size']}")
    print("## CPU, sherpa-onnx (onnx_results.jsonl)")
    table(load("onnx_results.jsonl"), lambda r: f"{r['model']} {r['threads']} threads")
    print("## batch 1 (bs1.jsonl)")
    table(load("bs1.jsonl"), lambda r: f"{r['model']} {r['dtype']} bs{r['batch_size']}")


if __name__ == "__main__":
    main()
