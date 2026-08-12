#!/usr/bin/env python3
"""Mirror the identity-study data to a HuggingFace dataset repo.

Uploads the curated out/studies/identity tree: the fullv4 per-seed trial
files, pooled analyses, gates, preflight, provenance, quality and annotation
sidecars, and the contrastive activation dumps. Skips quarantine/ (invalidated
data) and logs. Idempotent: re-running uploads only changed files.

    uv run python scripts/hf_upload.py [--repo user/name] [--public]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "out" / "studies" / "identity"

README = """\
# Identity Introspection Pipeline — data

Data for a WeirdChat-seeded introspection study on Qwen3.6-27B: does a
steered model report the intervention applied to its own residual stream
better than context or its visible text would allow?

- `caseA_body/`, `caseB_ai/` — per-seed trial records (stamp `fullv4`),
  one JSON per WeirdChat prompt: both arms (W = seeded reply exhibits the
  behavior, N = does not), conditions free/none/sweep/decoy, full context
  windows, regulator actions, and base-rate-corrected letter readouts
  (cells CTX / TXT / JS).
- `analysis_case{A,B}_fullv4.json` — pooled per-arm regressions and Delta.
- `gates/`, `preflight/` — G1/G2 gate runs and instrument preflights.
- `contrastive/` — residual-stream activations (safetensors) for every
  WeirdChat W/N pair: keys `mean` and `last`, shape
  [n_pairs, 2 (W,N), n_layers, hidden]; sidecar JSON maps rows to prompt
  ids and layers; `g1_contrastive_*.json` holds leave-one-out separation.
- `PROVENANCE.json`, `quality_fullv4.json`, `annotations_fullv4.json` —
  model receipts per stamp, content-quality verdicts, WeirdChat judge
  annotations joined back onto the seeds.

Subject model: `qwen/qwen3.6-27b` (WeirdChat transcripts); measured model:
`Qwen/Qwen3.6-27B`; steering via a Jacobian-lens token-set direction at
layer 42 (see the paper for the negative result this documents).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="unrulyabstractions/identity-introspection-weirdchat")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public,
                    exist_ok=True)
    (SRC / "README.md").write_text(README)
    api.upload_folder(
        repo_id=args.repo, repo_type="dataset", folder_path=str(SRC),
        commit_message="Mirror out/studies/identity",
        ignore_patterns=["quarantine/**", "*.log", "*.prev*", "shard_*",
                         "nohup_*"])
    files = api.list_repo_files(args.repo, repo_type="dataset")
    print(f"uploaded; repo now has {len(files)} files:")
    for f in sorted(files):
        print(" ", f)


if __name__ == "__main__":
    main()
