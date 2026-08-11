#!/usr/bin/env bash
#
# sync_up.sh — push this repo UP to the Vast.ai box (local -> cloud).
#
# The ONLY direction that writes to the remote (adapted from
# work/apart/jun2026/cloud/sync_up.sh). Pulling results lives in sync_back.sh.
#
# Usage:
#   bash cloud/sync_up.sh
#   DRY_RUN=1 bash cloud/sync_up.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
INSTANCE_FILE="$HERE/.vast_instance_id"
REMOTE_ROOT="${REMOTE_ROOT:-/root/apart-aug2026}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

INSTANCE="${INSTANCE:-}"
if [ -z "$INSTANCE" ]; then
  [ -f "$INSTANCE_FILE" ] || { echo "No instance id; run cloud/vast_launch.sh" >&2; exit 1; }
  INSTANCE="$(cat "$INSTANCE_FILE")"
fi

. "$HERE/_ssh_target.sh"
_resolve_ssh_target || exit 1

RSYNC_E="ssh $SSH_EPHEMERAL_OPTS -i $SSH_KEY -p $SSH_PORT"
DRY=""; [ "${DRY_RUN:-0}" = "1" ] && DRY="--dry-run"

# Retry on transient drops; --partial resumes instead of restarting.
rsync_retry() {
  local n=0
  until rsync -ah $DRY --timeout=120 --partial -e "$RSYNC_E" "$@"; do
    n=$((n+1)); [ "$n" -ge 6 ] && { echo "[sync_up] rsync FAILED after 6 tries" >&2; return 1; }
    echo "[sync_up] rsync retry $n/6 (transient drop); resuming ..." >&2; sleep 8
  done
}

# Stamp the revision being pushed. The box gets no .git (excluded below), so
# io_utils.git_rev() cannot shell out to git there and every artifact a remote
# run writes would say "unknown" — unattributable to any code version. Several
# result-changing fixes landed mid-campaign, so that link is load-bearing.
# "-dirty" is appended when the working tree has uncommitted changes: the stamp
# must never imply the pushed code equals that commit when it does not.
REV="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
  REV="${REV}-dirty"
fi
printf '%s\n' "$REV" > "$REPO_ROOT/.git_rev"
echo "[sync_up] stamping .git_rev = $REV"

# data/ IS included — the synthetic prompt sets are pipeline input. runs/ and
# sync/ stay local; no --delete so the remote is never surprise-pruned.
echo "[sync_up] code  $REPO_ROOT/  ->  $SSH_HOST:$REMOTE_ROOT/"
rsync_retry \
  --exclude='/runs/'          \
  --exclude='/out/'           \
  --exclude='/paper/build/'   \
  --exclude='/third_party/'   \
  --exclude='/hf_cache/'      \
  --exclude='/sync/'          \
  --exclude='/.backups/'      \
  --exclude='/cloud/.vast_instance_id' \
  --exclude='.git/'           \
  --exclude='.venv/'          \
  --exclude='__pycache__/'    \
  --exclude='*.pyc'           \
  --exclude='.DS_Store'       \
  --exclude='.pytest_cache/'  \
  --exclude='.ruff_cache/'    \
  --exclude='*.egg-info/'     \
  "$REPO_ROOT/" "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/"

echo "[sync_up] done."
echo "[sync_up] next: bash cloud/at_vast.sh 'bash cloud/at_setup.sh'"
