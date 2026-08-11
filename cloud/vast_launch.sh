#!/usr/bin/env bash
#
# vast_launch.sh — search Vast.ai for a matching GPU offer, create an instance,
# poll until running, and record the id in cloud/.vast_instance_id.
#
# Adapted from work/apart/jun2026/cloud/vast_launch.sh. Costs money once it
# creates an instance; nothing else here spends.
#
# Prereqs: `pip install vastai`, `vastai set api-key <KEY>`, SSH key on the account.
#
# Usage:
#   bash cloud/vast_launch.sh                     # interactive confirm
#   YES=1 GPU_NAME=RTX_4090 MAX_PRICE=0.60 bash cloud/vast_launch.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ----------------------------- CONFIG -----------------------------
# Default: one RTX 4090 (24 GB) — enough for ModernBERT fine-tuning and a ~9B
# fp16 residual-stream model on short prompts.
GPU_NAME="${GPU_NAME:-RTX_4090}"
NUM_GPUS="${NUM_GPUS:-1}"
DISK="${DISK:-60}"            # GB; permanent after creation
MAX_PRICE="${MAX_PRICE:-0.60}"
# Minimum per-GPU VRAM in GB. CRITICAL: a GPU_NAME like A100_PCIE spans BOTH
# 40 GB and 80 GB variants, and the cheapest-first order picks the 40 GB one —
# which silently under-provisions (a 32B bf16 model needs ~64 GB and OOMs on
# load). Always set MIN_VRAM for a large model. Default 0 keeps old behaviour.
MIN_VRAM="${MIN_VRAM:-0}"     # GB per GPU
# Minimum host reliability. ORDER defaults to cheapest-first, and the cheapest
# 80 GB offers are routinely the least reliable ones (0.55/hr at reliability
# 0.78 sitting above a 0.87/hr at 0.998). A host that dies mid-run costs far
# more than the few cents saved, because the generation pass restarts from zero.
# Default 0 keeps old behaviour; set 0.98 for anything long or expensive.
MIN_REL="${MIN_REL:-0}"
# Minimum host download bandwidth (Mbps). THE decisive lesson from the causal
# runs: low-inet_down hosts stall HF/Xet downloads at ~0 MB/s and the box bills
# while nothing transfers. A model that loads in 1 min on a 12000 Mbps 4090 hangs
# for an hour on a 400 Mbps host. Filtered in the query DSL (inet_down is a valid
# field). Default 0 keeps old behaviour; set 3000+ for anything that downloads.
MIN_INET="${MIN_INET:-0}"     # Mbps
ORDER="${ORDER:-dph_total+}"  # cheapest first
IMAGE="${IMAGE:-vastai/pytorch:@vastai-automatic-tag}"
PORTAL_ENV='-p 1111:1111 -p 8080:8080 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" -e DATA_DIRECTORY="/workspace/"'
# ------------------------------------------------------------------

command -v vastai >/dev/null || { echo "vastai not found. Run: pip install vastai" >&2; exit 1; }

# gpu_ram in Vast's QUERY DSL is in GB (the --raw OUTPUT reports MB — do not
# confuse them; multiplying by 1024 here asks for "80896 GB" and matches nothing).
VRAM_CLAUSE=""
[ "${MIN_VRAM}" -gt 0 ] 2>/dev/null && VRAM_CLAUSE="gpu_ram>=${MIN_VRAM}"
INET_CLAUSE=""
[ "${MIN_INET}" -gt 0 ] 2>/dev/null && INET_CLAUSE="inet_down>=${MIN_INET}"
QUERY="gpu_name=${GPU_NAME} num_gpus>=${NUM_GPUS} ${VRAM_CLAUSE} ${INET_CLAUSE} verified=true rentable=true direct_port_count>=1 disk_space>=${DISK} dph_total<=${MAX_PRICE}"
echo ">> Searching offers: ${QUERY}  (then MIN_REL>=${MIN_REL} applied here, not in the query)"
OFFERS_JSON="$(vastai search offers "$QUERY" -o "$ORDER" --raw)"

# MIN_REL is filtered HERE, not in the query DSL. Vast rejects a `reliability2`
# clause with "Unrecognized field" and then silently returns the UNFILTERED list,
# so putting it in the query looks like a guard while enforcing nothing. The
# reliability figure is present in the --raw output, so we filter the JSON.
OFFER_ID="$(MIN_REL="$MIN_REL" printf '%s' "$OFFERS_JSON" | MIN_REL="$MIN_REL" python3 -c '
import sys, json, os
mr = float(os.environ.get("MIN_REL", "0") or 0)
offers = json.load(sys.stdin) or []
kept = [o for o in offers if float(o.get("reliability2") or 0) >= mr]
if mr > 0:
    print("   %d of %d offers meet reliability>=%.3f" % (len(kept), len(offers), mr),
          file=sys.stderr)
    for o in offers:
        if o not in kept:
            print("   rejected %s at $%.3f/hr, reliability %.3f" % (
                o.get("id"), o.get("dph_total", 0.0),
                float(o.get("reliability2") or 0)), file=sys.stderr)
print(kept[0]["id"] if kept else "")
')"
[[ -n "$OFFER_ID" ]] || { echo "No offer met GPU_NAME/MAX_PRICE/MIN_REL; loosen one." >&2; exit 1; }

OFFER_ID="$OFFER_ID" printf '%s' "$OFFERS_JSON" | OFFER_ID="$OFFER_ID" python3 -c '
import sys, json, os
oid = int(os.environ["OFFER_ID"])
o = next(x for x in json.load(sys.stdin) if x.get("id") == oid)
print(">> Chosen offer: %s | %sx %s | $%.3f/hr | reliability=%.3f | net_down=%sMbps" % (
    o.get("id"), o.get("num_gpus"), o.get("gpu_name"),
    o.get("dph_total", 0.0), o.get("reliability2", 0.0), o.get("inet_down")))
'

if [[ "${YES:-0}" != "1" ]]; then
  read -r -p ">> Create this instance? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 0; }
fi

CREATE_JSON="$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --env "$PORTAL_ENV" \
  --onstart-cmd 'entrypoint.sh' \
  --disk "$DISK" \
  --ssh --direct \
  --raw)"
INSTANCE_ID="$(printf '%s' "$CREATE_JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("new_contract",""))')"
[[ -n "$INSTANCE_ID" ]] || { echo ">> Create failed: $CREATE_JSON" >&2; exit 1; }

# Crash-safe: record the id BEFORE polling so a failure here never orphans a
# billing box — vast_destroy.sh can always find it.
echo "$INSTANCE_ID" > "$HERE/.vast_instance_id"
echo ">> Created instance ${INSTANCE_ID}; waiting for it to run..."

for i in $(seq 1 80); do
  # instances-v1 paginates (25/page); walk every page or a new instance sitting
  # on page 2 reads as "missing" forever and the poll never sees it run.
  STATUS="$(INSTANCE_ID="$INSTANCE_ID" python3 -c '
import sys, json, os, subprocess
iid = int(os.environ["INSTANCE_ID"])
token = None; status = "missing"
for _ in range(40):
    cmd = ["vastai", "show", "instances-v1", "--raw"]
    if token:
        cmd += ["--next-token", token]
    try:
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    except Exception:
        break
    rows = d if isinstance(d, list) else d.get("instances", d.get("results", []))
    hit = next((r for r in rows if isinstance(r, dict) and r.get("id") == iid), None)
    if hit is not None:
        status = hit.get("actual_status") or hit.get("cur_state") or "unknown"
        break
    token = d.get("next_token") if isinstance(d, dict) else None
    if not token or not rows:
        break
print(status)
')"
  echo "   [$i] status: ${STATUS}"
  [[ "$STATUS" == "running" ]] && { echo ">> Instance is running."; break; }
  sleep 15
done

echo ">> Next: bash cloud/sync_up.sh && bash cloud/at_vast.sh 'bash cloud/at_setup.sh'"
echo ">> When finished: bash cloud/vast_destroy.sh --yes-i-am-really-sure"
