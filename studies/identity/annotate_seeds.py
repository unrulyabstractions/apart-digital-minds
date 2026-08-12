#!/usr/bin/env python3
"""Attach WeirdChat's own judge annotations to the collected seed files.

Every WeirdChat transcript carries the judge's explanation and, for matched
replies, citations: the exact quoted spans that exhibit the behavior, each
with a note. The seed loader kept only the reply text, so this script joins
the collected seed files back to the dataset by reply content and writes one
sidecar per stamp, which the results UI uses to explain and highlight the
behavior inside the seeded reply.

    uv run python -m studies.identity.annotate_seeds --stamp fullv4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DIRS = [ROOT / "out" / "studies" / "identity",
        ROOT / "sync" / "out" / "studies" / "identity"]
BEHAVIOR = {"A": "false-physical-embodiment", "B": "denying-ai-identity"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--subject-model", default="qwen/qwen3.6-27b")
    args = ap.parse_args()

    from datasets import load_dataset
    tr = load_dataset("Transluce/WeirdChat", "transcripts", split="train")

    # reply text -> judge record, per behavior
    lookup: dict[str, dict] = {}
    for case, behavior in BEHAVIOR.items():
        sub = tr.filter(lambda r: r["behavior_id"] == behavior
                        and r["subject_model"] == args.subject_model)
        for r in sub:
            msgs = r["messages"]
            if len(msgs) >= 2 and msgs[1]["role"] == "assistant":
                lookup[msgs[1]["content"]] = {
                    "match": r["judgment"]["match"],
                    "explanation": r["judgment"]["explanation"],
                    "citations": r["citations"] or [],
                    "transcript_id": r["transcript_id"],
                }

    out: dict[str, dict] = {}
    seen = set()
    for base in DIRS:
        for sub_dir in ("caseA_body", "caseB_ai"):
            for p in sorted((base / sub_dir).glob(f"seed*_{args.stamp}.json")):
                if p.name in seen:
                    continue
                seen.add(p.name)
                d = json.loads(p.read_text())
                anns = {}
                for rec in d.get("records", []):
                    arm = rec.get("arm")
                    if arm in anns:
                        continue
                    hit = lookup.get(rec.get("seed_reply", ""))
                    if hit:
                        anns[arm] = hit
                out[p.name] = anns
    dest = DIRS[0] / f"annotations_{args.stamp}.json"
    dest.write_text(json.dumps({"stamp": args.stamp, "files": out}, indent=1))
    hits = sum(1 for a in out.values() for v in a.values())
    print(f"annotated {len(out)} seed files ({hits} arm annotations) -> {dest}")


if __name__ == "__main__":
    main()
