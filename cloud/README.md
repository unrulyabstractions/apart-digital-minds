# Running the identity pipeline on Vast.ai (Qwen3.6-27B)

The 27B does not fit the 48 GB Mac (~54 GB weights). It is also the real
WeirdChat subject model, so its own transcripts are the seeds. Rent one
A100-80GB, run both cases, pull the results, and tear down only after every
file is proven captured.

Prereqs: `pip install vastai`, `vastai set api-key <KEY>`, SSH key on the
account and at `~/.ssh/id_ed25519`.

```bash
# 1. Launch an 80 GB, reliable, fast-download box (~ $1-1.5/hr).
YES=1 GPU_NAME=A100_SXM4 MIN_VRAM=80 MIN_REL=0.98 MIN_INET=3000 \
  DISK=120 MAX_PRICE=1.80 bash cloud/vast_launch.sh

# 2. Push the code, set up the env, pre-fetch the model + lens.
bash cloud/sync_up.sh
bash cloud/at_vast.sh "bash cloud/at_setup.sh"

# 3. Run both cases at scale (auto layer, full permutations).
bash cloud/at_vast.sh "STAMP=run1 NSEEDS=40 bash cloud/at_run.sh"

# 4. Pull results down into sync/ and inspect.
bash cloud/sync_back.sh

# 5. Tear down — only after every remote file is proven captured by bytes.
bash cloud/capture_and_destroy.sh --yes-i-am-really-sure
```

`capture_and_destroy.sh` refuses to destroy unless every file under the remote
`out/` has a byte-identical local copy. If it refuses, the box keeps billing
(recoverable); it never deletes the only copy of a run (not recoverable).

Iterate by editing locally, `sync_up.sh`, and re-running `at_run.sh` with a new
`STAMP`. The runner never overwrites a prior stamp's files.
