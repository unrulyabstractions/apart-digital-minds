#!/usr/bin/env bash
#
# capture_and_destroy.sh — the ONLY sanctioned teardown. It PROVES every remote
# file is on local disk, by bytes, before it will destroy the box.
#
#   INSTANCE=<id> bash cloud/capture_and_destroy.sh --yes-i-am-really-sure
#
# Steps:
#   1. Enumerate every file under the remote result roots (size + path), no
#      extension or directory filter beyond provably-reproducible caches.
#   2. rsync those results down into sync/ (new files only; never clobbers).
#   3. Re-enumerate and compare BYTES: every remote file must have a local copy
#      of identical size. Any mismatch or missing file REFUSES the destroy.
#   4. Only then call vast_destroy.sh.
#
# Nothing here deletes anything local. If capture cannot be proven, the box is
# left running and billing — a recoverable cost; deleting the only copy is not.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/root/apart-aug2026}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
LOCAL_SYNC="$REPO_ROOT/sync"
# Result roots to sweep. Add roots here if a run writes elsewhere.
ROOTS="${ROOTS:-$REMOTE_ROOT/out $REMOTE_ROOT/.git_rev}"

[ "${1:-}" == "--yes-i-am-really-sure" ] || {
  echo "Refusing: pass --yes-i-am-really-sure (destroy is irreversible)." >&2
  exit 2
}

INSTANCE="${INSTANCE:-}"
if [ -z "$INSTANCE" ]; then
  [ -f "$HERE/.vast_instance_id" ] || { echo "No instance id." >&2; exit 2; }
  INSTANCE="$(cat "$HERE/.vast_instance_id")"
fi

. "$HERE/_ssh_target.sh"
_resolve_ssh_target || exit 1
SSH="ssh $SSH_EPHEMERAL_OPTS -i $SSH_KEY -p $SSH_PORT $SSH_USER@$SSH_HOST"

echo "[capture] 1/3 enumerate remote files under: $ROOTS"
REMOTE_LIST="$($SSH "find $ROOTS -type f -printf '%s\t%p\n' 2>/dev/null" || true)"
[ -n "$REMOTE_LIST" ] || { echo "[capture] REFUSED: no remote files found; not destroying." >&2; exit 1; }
N_REMOTE="$(printf '%s\n' "$REMOTE_LIST" | grep -c . || true)"
echo "[capture] $N_REMOTE remote files"

echo "[capture] 2/3 pull into $LOCAL_SYNC (new files only, no clobber)"
mkdir -p "$LOCAL_SYNC"
RSYNC_E="ssh $SSH_EPHEMERAL_OPTS -i $SSH_KEY -p $SSH_PORT"
rsync -ah --timeout=120 --partial --ignore-existing -e "$RSYNC_E" \
  "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/out/" "$LOCAL_SYNC/out/" || {
    echo "[capture] REFUSED: rsync failed; box untouched." >&2; exit 1; }

echo "[capture] 3/3 verify every remote file has a byte-identical local copy"
MISS="$(REMOTE_LIST="$REMOTE_LIST" REMOTE_ROOT="$REMOTE_ROOT" \
  LOCAL_SYNC="$LOCAL_SYNC" REPO_ROOT="$REPO_ROOT" python3 - <<'PYEOF'
import os
remote = os.environ["REMOTE_LIST"].strip().splitlines()
rroot = os.environ["REMOTE_ROOT"].rstrip("/")
sync = os.environ["LOCAL_SYNC"]
repo = os.environ["REPO_ROOT"]
missing = []
for line in remote:
    if "\t" not in line:
        continue
    size, path = line.split("\t", 1)
    rel = path[len(rroot):].lstrip("/")           # e.g. out/studies/identity/..
    # a local copy may sit in sync/ (fresh pull) or already promoted in the repo
    cands = [os.path.join(sync, rel), os.path.join(repo, rel)]
    if not any(os.path.exists(c) and os.path.getsize(c) == int(size) for c in cands):
        missing.append(f"{size}\t{path}")
print("\n".join(missing))
PYEOF
)"

if [ -n "$MISS" ]; then
  echo "[capture] REFUSED: these remote files are NOT captured (size mismatch or missing):" >&2
  printf '%s\n' "$MISS" >&2
  echo "[capture] The box is UNTOUCHED and still billing. Re-run sync, then retry." >&2
  exit 1
fi

echo "[capture] all $N_REMOTE remote files captured by bytes. Destroying $INSTANCE."
exec bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure
