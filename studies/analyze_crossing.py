#!/usr/bin/env python3
"""Was the readout a property of the window, or of the reader?

    python studies/analyze_crossing.py

Three things are read off the crossing:

    reader spread     how far apart two readers land on the identical window
    subject spread    how far apart one reader lands across different subjects
    same weights      whether a reader made of the subject's own weights
                      differs from a foreign reader on the same window

If reader spread is small next to subject spread, the readout is carried by the
window and the instrument is interchangeable. If it is large, the earlier
results measured the reader.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dminds import paths  # noqa: E402

HOME = paths.OUT / "studies" / "crossing"


def distance(a: dict, b: dict) -> float:
    """Total variation distance between two readouts."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, width=7) -> str:
    return f"{value:.3f}".rjust(width) if value is not None else "      .".rjust(width)


def short(spec: str) -> str:
    return spec.split("/")[-1].replace("-Instruct-2507", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HOME / "summary.json"))
    args = parser.parse_args()

    rows = json.loads((HOME / "crossing.json").read_text())
    meta = json.loads((HOME / "meta.json").read_text())
    readers, subjects = meta["readers"], meta["subjects"]

    # One readout per (reader, subject, probe, scenario, turn).
    at = {(r["reader"], r["subject"], r["probe"], r["scenario"], r["turn"]): r
          for r in rows}
    cells = {(r["subject"], r["probe"], r["scenario"], r["turn"]) for r in rows}

    print(f"\n{'=' * 76}\nreader spread: two readers, one window\n{'=' * 76}")
    print(f"  {'probe':<18}{'reader pair':<34}{'distance':>10}{'agree':>9}")
    spread = {}
    for probe in meta["probes"]:
        for a, b in combinations(readers, 2):
            gaps, agree = [], []
            for (subject, p, scenario, turn) in cells:
                if p != probe:
                    continue
                ra = at.get((a, subject, p, scenario, turn))
                rb = at.get((b, subject, p, scenario, turn))
                if ra and rb:
                    gaps.append(distance(ra["probs"], rb["probs"]))
                    agree.append(ra["label"] == rb["label"])
            spread.setdefault(probe, []).extend(gaps)
            rate = f"{100 * sum(agree) / len(agree):.0f}%" if agree else "."
            print(f"  {probe:<18}{short(a) + ' vs ' + short(b):<34}"
                  f"{fmt(mean(gaps)):>10}{rate:>9}")

    print(f"\n{'=' * 76}\nsubject spread: one reader, different subjects\n{'=' * 76}")
    print(f"  {'probe':<18}{'reader':<34}{'distance':>10}")
    subject_spread = {}
    for probe in meta["probes"]:
        for reader in readers:
            gaps = []
            for s1, s2 in combinations(subjects, 2):
                for (subject, p, scenario, turn) in cells:
                    if p != probe or subject != s1:
                        continue
                    r1 = at.get((reader, s1, p, scenario, turn))
                    r2 = at.get((reader, s2, p, scenario, turn))
                    if r1 and r2:
                        gaps.append(distance(r1["probs"], r2["probs"]))
            subject_spread.setdefault(probe, []).extend(gaps)
            print(f"  {probe:<18}{short(reader):<34}{fmt(mean(gaps)):>10}")

    print(f"\n{'=' * 76}\nwhat carries the readout\n{'=' * 76}")
    print(f"  {'probe':<18}{'reader spread':>15}{'subject spread':>16}"
          f"{'carried by':>14}")
    verdict = {}
    for probe in meta["probes"]:
        r, s = mean(spread.get(probe, [])), mean(subject_spread.get(probe, []))
        who = "." if r is None or s is None else (
            "the window" if s > r * 1.5 else
            "the reader" if r > s * 1.5 else "both"
        )
        verdict[probe] = {"reader_spread": r, "subject_spread": s, "carried_by": who}
        print(f"  {probe:<18}{fmt(r):>15}{fmt(s):>16}{who:>14}")

    print(f"\n{'=' * 76}\nsame weights as the subject, against a foreign reader"
          f"\n{'=' * 76}")
    print("  A reader made of the subject's weights only counts as privileged if")
    print("  it sits further from a foreign reader than two foreign readers sit")
    print("  from each other on the same window.\n")
    print(f"  {'probe':<18}{'own vs foreign':>16}{'foreign vs foreign':>20}"
          f"{'privileged':>13}")
    privileged = {}
    for probe in meta["probes"]:
        own_gaps, foreign_gaps = [], []
        for subject in subjects:
            others = [r for r in readers if r != subject]
            for (subj, p, scenario, turn) in cells:
                if p != probe or subj != subject:
                    continue
                here = {r: at.get((r, subject, p, scenario, turn)) for r in readers}
                if subject in readers and here.get(subject):
                    own_gaps += [
                        distance(here[subject]["probs"], here[o]["probs"])
                        for o in others if here.get(o)
                    ]
                foreign_gaps += [
                    distance(here[a]["probs"], here[b]["probs"])
                    for a, b in combinations(others, 2)
                    if here.get(a) and here.get(b)
                ]
        own, foreign = mean(own_gaps), mean(foreign_gaps)
        verdict_here = "." if own is None or foreign is None else (
            "yes" if own > foreign * 1.25 else "no"
        )
        privileged[probe] = {"own_vs_foreign": own, "foreign_vs_foreign": foreign,
                             "privileged": verdict_here}
        print(f"  {probe:<18}{fmt(own):>16}{fmt(foreign):>20}{verdict_here:>13}")

    summary = {"verdict": verdict, "privileged_access": privileged}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
