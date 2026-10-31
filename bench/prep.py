"""Build asr-ja-bench subsets (fleurs / jsut / jvnv) from the sample lists.

Reads data/<subset>/samples.jsonl, applies the published patches, checks the
transcript hashes, and writes <out>/<subset>.jsonl + 16 kHz FLAC audio.

python -I prep.py --data bench-repo/data --out bench --subset fleurs [--jsut zip] [--jvnv zip]
"""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

SR = 16000
FLEURS_REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def patch(t, p):
    for s, e, r in reversed(p or []):
        t = t[:s] + r + t[e:]
    return t


def decode(data):
    info = sf.info(io.BytesIO(data))
    if info.samplerate == SR and info.channels == 1:
        return sf.read(io.BytesIO(data), dtype="float32")[0]
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "-", "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
        input=data, capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype="float32")


def fleurs():
    tsv = hf_hub_download("google/fleurs", "data/ja_jp/test.tsv", repo_type="dataset",
                          revision=FLEURS_REV)
    texts = {}
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            texts[row[1].removesuffix(".wav")] = row[3]
    tgz = hf_hub_download("google/fleurs", "data/ja_jp/audio/test.tar.gz", repo_type="dataset",
                          revision=FLEURS_REV)
    out = {}
    with tarfile.open(tgz) as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".wav"):
                uid = m.name.rsplit("/", 1)[-1].removesuffix(".wav")
                if uid in texts:
                    out[uid] = (texts[uid], lambda d=tf.extractfile(m).read(): d)
    return out


def jsut(path):
    z = zipfile.ZipFile(path)
    r = "jsut_ver1.1/basic5000/"
    out = {}
    for line in z.read(r + "transcript_utf8.txt").decode().splitlines():
        if line.strip():
            uid, text = line.split(":", 1)
            out[uid] = (text, lambda u=uid: z.read(f"{r}wav/{u}.wav"))
    return out


def jvnv(path):
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    scripts = {}
    for line in z.read("jvnv_v1/transcription.csv").decode().splitlines():
        if line.strip():
            key, _, text = line.split("|")
            scripts[key] = text.strip()
    out = {}
    for spk in ("F1", "F2", "M1", "M2"):
        for key, text in scripts.items():
            emo, sess, _ = key.split("_")
            n = f"jvnv_v1/{spk}/{emo}/{sess}/{spk}_{key}.wav"
            if n in names:
                out[f"{spk}_{key}"] = (text, lambda n=n: z.read(n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset", action="append", required=True)
    ap.add_argument("--jsut", type=Path)
    ap.add_argument("--jvnv", type=Path)
    a = ap.parse_args()
    for subset in a.subset:
        up = {"fleurs": fleurs, "jsut": lambda: jsut(a.jsut), "jvnv": lambda: jvnv(a.jvnv)}[subset]()
        samples = [json.loads(x) for x in open(a.data / subset / "samples.jsonl") if x.strip()]
        (a.out / "audio" / subset).mkdir(parents=True, exist_ok=True)
        rows, bad = [], 0
        for s in samples:
            if s["id"] not in up or sha(up[s["id"]][0]) != s["upstream_sha256"]:
                bad += 1
                continue
            text = patch(up[s["id"]][0], s.get("patch"))
            if sha(text) != s["text_sha256"]:
                bad += 1
                continue
            x = decode(up[s["id"]][1]())
            rel = f"audio/{subset}/{s['id']}.flac"
            sf.write(a.out / rel, x, SR, format="FLAC", subtype="PCM_16")
            rows.append({"id": s["id"], "audio": rel, "text": text, "duration": len(x) / SR})
        with open(a.out / f"{subset}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{subset}] {len(rows)}/{len(samples)} ok, {bad} problems", flush=True)


if __name__ == "__main__":
    main()
