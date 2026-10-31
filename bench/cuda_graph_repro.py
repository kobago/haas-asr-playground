"""Reproduce the crash of NeMo's CUDA-graph TDT decoder at batch >= 32 (RTX 2060 SUPER).

python cuda_graph_repro.py BENCH_DIR bf16|fp16|fp32 BATCH [START]
NOGRAPH=1 python cuda_graph_repro.py ...   # with the CUDA-graph decoder off: no crash

With the decoder on, "CUDA error: an illegal memory access was encountered"
around FLEURS batch 96 (batch 32, bf16 and fp16).
"""
import json, sys, numpy as np, soundfile as sf, torch, haas_asr
from pathlib import Path
bench = Path(sys.argv[1]); dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[sys.argv[2]]
bs = int(sys.argv[3]); start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rows = [json.loads(x) for x in open(bench / "fleurs.jsonl")]
m = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", dtype=dt)
import os
if os.environ.get("NOGRAPH"):
    from omegaconf import open_dict
    cfg = m.cfg.decoding
    with open_dict(cfg): cfg.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(cfg, verbose=False); print("cuda graphs off", flush=True)
for i in range(start, len(rows), bs):
    b = [sf.read(bench / r["audio"], dtype="float32")[0] for r in rows[i:i+bs]]
    print("batch", i, "maxdur", max(len(x) for x in b) / 16000, flush=True)
    m.transcribe(b, batch_size=len(b), verbose=False); torch.cuda.synchronize()
print("ALL OK")
