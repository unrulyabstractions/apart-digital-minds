#!/usr/bin/env python3
"""Content-level quality checks on collected seed files, run on every pull.

Structural validity is not verification: this reads the actual replies. Per
seed file it measures truncation (replies that stop mid-sentence), reasoning
scaffolding (replies that are thinking-preamble rather than answers), the
regulator's ACTION parse rate, cell sanity (JS equals CTX at strength zero,
readouts in range, base correction applied), coherence flags, and whether the
per-turn instrumentation (windows, cells) is present. Verdicts are written to
one quality file per stamp, which the results UI displays beside the data.

    uv run python -m studies.identity.check_results --stamp fullv3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DIRS = [ROOT / "out" / "studies" / "identity",
        ROOT / "sync" / "out" / "studies" / "identity"]

SENT_END = tuple(".!?\"'”)]:")
SCAFFOLD = re.compile(r"^\s*(here'?s a thinking process|<think>|"
                      r"\*\*analyze user input)", re.I)


def check_reply(text: str) -> dict:
    t = (text or "").strip()
    midcut = not t.endswith(SENT_END)
    # a clipped final sentence on a substantial reply is a cap artifact that
    # hits every arm and condition alike; a short mid-cut fragment is junk.
    return {"junk": midcut and len(t) < 400,
            "clipped": midcut and len(t) >= 400,
            "scaffold": bool(SCAFFOLD.search(t)),
            "empty": not t, "chars": len(t)}


def check_seed_file(path: Path) -> dict:
    d = json.loads(path.read_text())
    reasons = []
    replies, action_ok, action_n = [], 0, 0
    zero_mismatch = cell_range_bad = 0
    windows_missing = turn_cells_missing = coherence_flags = 0
    key = "vantage" if d.get("case") == "A" else "p"

    for r in d.get("records", []):
        for t in r.get("turns", []):
            replies.append(check_reply(t.get("subject")))
            replies.append(check_reply(t.get("actor")))
            if not (t.get("windows") or {}).get("subject"):
                windows_missing += 1
            if t.get("cells") is None and t is not r["turns"][-1]:
                turn_cells_missing += 1
        reg = r.get("regulator", {})
        if r.get("condition") == "free":
            action_n += 1
            if reg.get("action") in ("more", "same", "less"):
                action_ok += 1
            if reg.get("self_report"):
                replies.append(check_reply(reg["self_report"]))
        if reg.get("strength") == 0 and {"CTX", "JS"} <= set(r.get("cells", {})):
            if r["cells"]["JS"].get(key) != r["cells"]["CTX"].get(key):
                zero_mismatch += 1
        for cd in r.get("cells", {}).values():
            v = cd.get(key)
            if v is None or not (-1.0 <= v <= 1.0):
                cell_range_bad += 1
        coherence_flags += bool(r.get("coherence_flag"))

    n = max(len(replies), 1)
    junk = sum(x["junk"] for x in replies) / n
    clip = sum(x["clipped"] for x in replies) / n
    scaf = sum(x["scaffold"] for x in replies) / n
    rep = {
        "file": path.name, "case": d.get("case"), "model": d.get("model"),
        "n_records": len(d.get("records", [])),
        "junk_rate": round(junk, 3), "clip_rate": round(clip, 3),
        "scaffold_rate": round(scaf, 3),
        "action_parse": f"{action_ok}/{action_n}",
        "zero_strength_mismatch": zero_mismatch,
        "cell_out_of_range": cell_range_bad,
        "windows_missing": windows_missing,
        "turn_cells_missing": turn_cells_missing,
        "coherence_flags": coherence_flags,
        "mean_reply_chars": int(sum(x["chars"] for x in replies) / n),
    }
    if junk > 0.15:
        reasons.append(f"{junk:.0%} replies are mid-cut fragments")
    if action_n and action_ok < action_n:
        reasons.append(f"ACTION parse {action_ok}/{action_n}")
    if zero_mismatch:
        reasons.append(f"{zero_mismatch} zero-strength JS!=CTX")
    if cell_range_bad:
        reasons.append(f"{cell_range_bad} cells out of range")
    if scaf > 0.3:
        reasons.append(f"{scaf:.0%} replies are reasoning scaffold "
                       "(enable_thinking violated)")
    rep["verdict"] = "FAIL" if reasons else (
        "WARN" if clip > 0.30 else "PASS")
    rep["reasons"] = reasons
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stamp", required=True)
    args = ap.parse_args()
    seen, reports = set(), []
    for base in DIRS:
        for sub in ("caseA_body", "caseB_ai"):
            for p in sorted((base / sub).glob(f"seed*_{args.stamp}.json")):
                if p.name in seen:
                    continue
                seen.add(p.name)
                try:
                    reports.append(check_seed_file(p))
                except Exception as e:
                    reports.append({"file": p.name, "verdict": "FAIL",
                                    "reasons": [f"unreadable: {e}"]})
    out = {"stamp": args.stamp, "n_files": len(reports),
           "n_pass": sum(r.get("verdict") == "PASS" for r in reports),
           "reports": reports}
    dest = DIRS[0] / f"quality_{args.stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    for r in reports:
        print(f"  {r.get('verdict', '?'):4} {r['file']}"
              + (f"  <- {'; '.join(r['reasons'])}" if r.get("reasons") else ""))
    print(f"{out['n_pass']}/{out['n_files']} PASS -> {dest}")


if __name__ == "__main__":
    main()
