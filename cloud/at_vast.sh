#!/usr/bin/env bash
#
# at_vast.sh — run a single command on the Vast.ai box over SSH.
#
# Thin "send a command, stream its output" wrapper (adapted from
# work/apart/jun2026/cloud/at_vast.sh). Pushing code is sync_up.sh; pulling
# results is sync_back.sh — the dangerous direction stays isolated there.
#
# Usage:
#   bash cloud/at_vast.sh "nvidia-smi"
#   INSTANCE=12345678 bash cloud/at_vast.sh "<cmd>"

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
INSTANCE_FILE="$HERE/.vast_instance_id"
REMOTE_ROOT="${REMOTE_ROOT:-/root/apart-aug2026}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

INSTANCE="${INSTANCE:-}"
if [ -z "$INSTANCE" ]; then
  [ -f "$INSTANCE_FILE" ] || { echo "No instance id; run cloud/vast_launch.sh" >&2; exit 1; }
  INSTANCE="$(cat "$INSTANCE_FILE")"
fi
[ $# -gt 0 ] || { echo "usage: bash cloud/at_vast.sh <command>" >&2; exit 1; }
CMD="$*"

. "$HERE/_ssh_target.sh"
_resolve_ssh_target || exit 1
SSH="ssh $SSH_EPHEMERAL_OPTS -i $SSH_KEY -p $SSH_PORT $SSH_USER@$SSH_HOST"

# cd only if the repo dir already exists (absent before the first sync_up).
# Redact inline secrets (OPENAI_API_KEY=..., HF_TOKEN=..., anything *KEY/ *TOKEN)
# before echoing — $CMD is frequently invoked with keys set inline, and the raw
# echo would leak them to the terminal, captured logs, and SSH server logs.
CMD_SAFE="$(printf '%s' "$CMD" | sed -E 's/([A-Za-z_]*(KEY|TOKEN|SECRET)[A-Za-z_]*=)[^ ]+/\1<redacted>/g')"
echo "[at_vast] run on $SSH_HOST:$SSH_PORT  ::  $CMD_SAFE"
exec $SSH "cd $REMOTE_ROOT 2>/dev/null || true; $CMD"
