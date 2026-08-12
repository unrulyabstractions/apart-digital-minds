#!/usr/bin/env bash
#
# launch_shard.sh — start one shard ON the box, robustly, and verify it took.
#
#   bash cloud/launch_shard.sh <STAMP> <OFFSET> <NSEEDS>
#
# Kills prior runs, purges bytecode, launches under setsid with nohup, writes a
# pidfile, then verifies the process survived 10 seconds and the log is
# growing. Exits nonzero (with the log tail) if the launch did not take, so a
# caller over ssh gets an honest verdict instead of a race.

set -u
cd "$(dirname "$0")/.."
STAMP="${1:?stamp}"; OFFSET="${2:?offset}"; NSEEDS="${3:-5}"

pkill -9 -f studies.identity 2>/dev/null
pkill -9 -f at_run 2>/dev/null
sleep 2
find studies src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
mkdir -p out/studies/identity
LOG="out/studies/identity/nohup_${STAMP}.log"
rm -f "$LOG"

setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  STAMP="$STAMP" CASES="A B" NSEEDS="$NSEEDS" OFFSET="$OFFSET" \
  SKIP=1 FULL=1 FORCE=1 CONDITIONS="free none sweep decoy" \
  nohup bash cloud/at_run.sh > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "out/studies/identity/shard_${STAMP}.pid"

sleep 10
if ! kill -0 "$PID" 2>/dev/null && ! pgrep -f studies.identity >/dev/null; then
  echo "LAUNCH FAILED — process died; log tail:"
  tail -15 "$LOG" 2>/dev/null
  exit 1
fi
echo "LAUNCH OK pid=$PID stamp=$STAMP offset=$OFFSET"
tail -2 "$LOG" 2>/dev/null
