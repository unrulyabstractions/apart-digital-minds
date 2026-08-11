#!/usr/bin/env bash
#
# at_run.sh — run the identity pipeline ON the box, full scale, both cases.
#
#   bash cloud/at_vast.sh "STAMP=run1 bash cloud/at_run.sh"
#
# Uses .venv/bin/python directly (uv run would undo the torch pin). Layer -1
# auto-picks ~2/3 depth from the 27B lens. Logs stream to out/studies/identity/.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="$PWD/.venv/bin/python"
[ -x "$PY" ] || { echo "[at_run] FATAL: run cloud/at_setup.sh first"; exit 1; }

STAMP="${STAMP:?set STAMP=<label>}"
MODEL="${MODEL:-hf:Qwen/Qwen3.6-27B}"
CASES="${CASES:-A B}"
NSEEDS="${NSEEDS:-40}"
CONDITIONS="${CONDITIONS:-free none sweep decoy}"
LAYER="${LAYER:--1}"

mkdir -p out/studies/identity
LOG="out/studies/identity/run_${STAMP}.log"
echo "[at_run] model=$MODEL cases=$CASES n=$NSEEDS layer=$LAYER stamp=$STAMP" | tee "$LOG"

HF_HUB_ENABLE_HF_TRANSFER=1 "$PY" -m studies.identity.run \
  --model "$MODEL" --cases $CASES --layer "$LAYER" \
  --n-seeds "$NSEEDS" --conditions $CONDITIONS \
  --full-perms --stamp "$STAMP" 2>&1 | tee -a "$LOG"

echo "[at_run] analysis"
"$PY" -m studies.identity.analyze --stamp "$STAMP" --case $CASES 2>&1 | tee -a "$LOG"
echo "[at_run] done; results under out/studies/identity/"
