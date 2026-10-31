#!/usr/bin/env bash
# Start the Haas playground on 0.0.0.0:7860 over HTTPS (Tailscale certificate).
set -euo pipefail
cd "$(dirname "$0")"
export TMPDIR="$HOME/.cache/haas-asr-eval/tmp"  # NeMo unpacks the .nemo here; /tmp is a small tmpfs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"
export HAAS_REPO="${HAAS_REPO:-$HOME/projects/github.com/Haas-ASR}"
CERTS="$HOME/.cache/haas-mic-app/certs"  # tailscale cert (see README.md)
# sherpa-onnx (ONNX models) is added on top of the Haas-ASR environment for this run only.
exec uv run --project "$HAAS_REPO" --group demo --with sherpa-onnx==1.13.6 python server.py \
    --host 0.0.0.0 --port 7860 --ssl-certfile "$CERTS/ts.crt" --ssl-keyfile "$CERTS/ts.key" "$@"
