#!/usr/bin/env bash
#
# at_setup.sh — environment setup that runs ON the Vast.ai box.
#
#   bash cloud/at_vast.sh "bash cloud/at_setup.sh"
#
# Installs uv, resolves the locked deps with the hf extra (torch, transformers,
# datasets, huggingface_hub), pins a CUDA-12.4 torch build (cu13x wheels fail
# CUDA init on pre-CUDA-13 hosts; the Vast fleet mixes both), and checks the GPU.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root on the remote

if ! command -v uv >/dev/null 2>&1; then
  echo "[at_setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[at_setup] uv sync --extra hf"
uv sync --extra hf

# Torch cu124 pin. WITH deps (never --no-deps): the torch wheel depends on the
# nvidia-*-cu12 pip packages for its CUDA runtime.
TORCH_PKGS="${TORCH_PKGS:-torch==2.6.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
echo "[at_setup] pinning $TORCH_PKGS via $TORCH_INDEX"
uv pip install --reinstall $TORCH_PKGS --index-url "$TORCH_INDEX"

# `uv run` re-syncs the venv to the lockfile and silently undoes the torch pin.
# Everything after setup must invoke .venv/bin/python DIRECTLY.
PY="$PWD/.venv/bin/python"
[ -x "$PY" ] || { echo "[at_setup] FATAL: $PY missing after uv sync"; exit 1; }

echo "[at_setup] device check"
"$PY" - <<'PYEOF'
import sys, torch
print("torch:", torch.__version__)
ok = torch.cuda.is_available()
if ok:
    p = torch.cuda.get_device_properties(0)
    print(f"cuda: {p.name}  {p.total_memory/1e9:.0f} GB")
else:
    print("[at_setup] FATAL: CUDA unavailable — refusing to run on CPU.")
    sys.exit(1)
PYEOF

echo "[at_setup] pre-fetching the 27B and its lens (this is the slow part) ..."
HF_HUB_ENABLE_HF_TRANSFER=1 "$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Qwen/Qwen3.6-27B")
hf_hub_download("neuronpedia/jacobian-lens",
                "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
print("model + lens cached")
PYEOF

echo "[at_setup] done. Run with .venv/bin/python (NOT uv run), e.g.:"
echo "  bash cloud/at_run.sh <stamp>"
