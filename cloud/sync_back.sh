#!/usr/bin/env bash
#
# sync_back.sh — pull remote outputs/ DOWN into a local quarantine dir (cloud -> local).
# (This repo writes results to outputs/, NOT runs/ — the rsync below targets
# outputs/. A prior version pulled runs/ and lost data; keep this in sync.)
#
# SAFETY (adapted verbatim from work/apart/jun2026/cloud/sync_back.sh):
#   1. Writes ONLY into the local gitignored sync/ directory — never outputs/ or code.
#   2. rsync --ignore-existing — existing local files are NEVER overwritten.
#   3. No --delete — nothing local is ever removed.
# Net effect: the worst the cloud can do is add new files under sync/.
# Inspect sync/ by hand, then promote what you trust into outputs/.
#
# Usage:
#   bash cloud/sync_back.sh
#   DRY_RUN=1 bash cloud/sync_back.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
INSTANCE_FILE="$HERE/.vast_instance_id"
REMOTE_ROOT="${REMOTE_ROOT:-/root/apart-aug2026}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
LOCAL_SYNC="$REPO_ROOT/sync"

INSTANCE="${INSTANCE:-}"
if [ -z "$INSTANCE" ]; then
  [ -f "$INSTANCE_FILE" ] || { echo "No instance id; run cloud/vast_launch.sh" >&2; exit 1; }
  INSTANCE="$(cat "$INSTANCE_FILE")"
fi

. "$HERE/_ssh_target.sh"
_resolve_ssh_target || exit 1

RSYNC_E="ssh $SSH_EPHEMERAL_OPTS -i $SSH_KEY -p $SSH_PORT"
DRY=""; [ "${DRY_RUN:-0}" = "1" ] && DRY="--dry-run"

mkdir -p "$LOCAL_SYNC"
echo "[sync_back] $SSH_HOST:$REMOTE_ROOT/outputs/  ->  $LOCAL_SYNC/  (new files only)"
rsync -ah $DRY --timeout=120 --partial --ignore-existing -e "$RSYNC_E" \
  "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/outputs/" "$LOCAL_SYNC/"

echo "[sync_back] done. Inspect $LOCAL_SYNC/ before promoting anything into outputs/."
