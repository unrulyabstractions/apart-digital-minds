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

# stale bytecode once ran old code after a sync (equal-second mtimes fool the
# pyc check); purge compiled caches so the synced source is what executes.
find studies src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null

PY="$PWD/.venv/bin/python"
[ -x "$PY" ] || { echo "[at_run] FATAL: run cloud/at_setup.sh first"; exit 1; }

# expandable segments keeps thousands of varying-length scoring passes from
# fragmenting the GPU into a spurious OOM (54 GB model leaves ~40 GB to work in).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="${STAMP:?set STAMP=<label>}"
MODEL="${MODEL:-hf:Qwen/Qwen3.6-27B}"
CASES="${CASES:-A B}"
NSEEDS="${NSEEDS:-40}"
CONDITIONS="${CONDITIONS:-free none sweep decoy}"
LAYER="${LAYER:--1}"
# Permutations: FULL=1 uses all 24 (the camera-ready pass); otherwise a capped
# subset (NPERMS, default 8) keeps an iteration cheap, which the plan allows.
FULL="${FULL:-0}"
NPERMS="${NPERMS:-8}"
PERM_FLAG="--n-perms $NPERMS"; [ "$FULL" = "1" ] && PERM_FLAG="--full-perms"
# FORCE=1 runs the trial phase even past failed gates (instrumented records;
# the gate files still stand and say what they say).
FORCE="${FORCE:-0}"
FORCE_FLAG=""; [ "$FORCE" = "1" ] && FORCE_FLAG="--force-trials"
SKIP="${SKIP:-0}"
[ "$SKIP" = "1" ] && FORCE_FLAG="$FORCE_FLAG --skip-gates"
OFFSET="${OFFSET:-0}"
FORCE_FLAG="$FORCE_FLAG --seed-offset $OFFSET"

mkdir -p out/studies/identity
LOG="out/studies/identity/run_${STAMP}.log"
echo "[at_run] model=$MODEL cases=$CASES n=$NSEEDS layer=$LAYER perms=$PERM_FLAG stamp=$STAMP" | tee "$LOG"

HF_HUB_ENABLE_HF_TRANSFER=1 "$PY" -m studies.identity.run \
  --model "$MODEL" --cases $CASES --layer "$LAYER" \
  --n-seeds "$NSEEDS" --conditions $CONDITIONS \
  $PERM_FLAG $FORCE_FLAG --stamp "$STAMP" 2>&1 | tee -a "$LOG"

echo "[at_run] analysis"
"$PY" -m studies.identity.analyze --stamp "$STAMP" --case $CASES 2>&1 | tee -a "$LOG"
echo "[at_run] done; results under out/studies/identity/"
