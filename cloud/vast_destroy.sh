#!/usr/bin/env bash
#
# vast_destroy.sh — destroy the recorded Vast.ai instance (stops billing).
#
# Usage:
#   bash cloud/vast_destroy.sh --yes-i-am-really-sure
#   INSTANCE=12345678 bash cloud/vast_destroy.sh --yes-i-am-really-sure

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
INSTANCE_FILE="$HERE/.vast_instance_id"

[ "${1:-}" == "--yes-i-am-really-sure" ] || {
  echo "This destroys the GPU instance and everything on it." >&2
  echo "usage: bash cloud/vast_destroy.sh --yes-i-am-really-sure" >&2
  exit 1
}

INSTANCE="${INSTANCE:-}"
if [ -z "$INSTANCE" ]; then
  [ -f "$INSTANCE_FILE" ] || { echo "No instance id recorded; nothing to destroy." >&2; exit 1; }
  INSTANCE="$(cat "$INSTANCE_FILE")"
fi

echo ">> Destroying instance $INSTANCE"
# The CLI prompts interactively and exits 0 even on "Aborted" — feed it the
# confirmation (printf, not `yes`: pipefail turns yes's SIGPIPE into exit 141)
# and verify the instance is actually gone before forgetting it.
printf 'y\n' | vastai destroy instance "$INSTANCE"
sleep 3
# instances-v1 PAGINATES (25/page); a single --raw page can miss the instance on
# a later page and falsely report it gone. Walk every page (mirrors _ssh_target.sh)
# before trusting that the box is really destroyed.
if INSTANCE="$INSTANCE" python3 -c "
import json, os, subprocess, sys
iid = int(os.environ['INSTANCE'])
token = None
for _ in range(40):  # hard page cap so a runaway token loop can never hang
    cmd = ['vastai', 'show', 'instances-v1', '--raw']
    if token:
        cmd += ['--next-token', token]
    try:
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    except Exception:
        sys.exit(1)  # cannot confirm gone -> treat as still-present, do not delete
    rows = d if isinstance(d, list) else d.get('instances', d.get('results', []))
    if any(r.get('id') == iid for r in rows):
        sys.exit(1)  # still listed
    token = d.get('next_token') if isinstance(d, dict) else None
    if not token or not rows:
        break
sys.exit(0)  # walked all pages, not found -> gone
"; then
  rm -f "$INSTANCE_FILE"
  echo ">> Destroyed and verified gone."
else
  echo ">> WARNING: instance $INSTANCE still listed — NOT removing $INSTANCE_FILE" >&2
  exit 1
fi
