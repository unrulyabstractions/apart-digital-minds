#!/usr/bin/env python3
"""Score the judge-free readings, with the unit of analysis set correctly.

The separation is computed over pairs of held-out scenarios, but the pairs are
not independent: they come from a smaller number of scenarios, and the five
turns inside a scenario are one conversation rather than five observations. So
the null shuffles which scenarios are pressure and which are positive, and the
p-value counts how often a relabelling separates them as well.

    python studies/analyze_subject_side.py

Reads out/studies/subject_side/*/summary.json and writes summary.json beside it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dminds import paths  # noqa: E402

HOME = paths.OUT / "studies" / "subject_side"


def auc(positive, negative) -> float | None:
    """Probability a positive scores above a negative, ties counted as half."""
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def permutation_p(pressure, positive, trials=20000, seed=0) -> float:
    """How often relabelling the scenarios separates them at least this well.

    Scenarios are the unit, so a permutation moves whole scenarios between the
    two sides. Shuffling turns instead would treat five readings of one
    conversation as five independent observations and report a p-value far
    smaller than the design earns.
    """
    observed = auc(pressure, positive)
    if observed is None:
        return 1.0
    pool = list(pressure) + list(positive)
    k = len(pressure)
    rng = random.Random(seed)
    at_least = 0
    for _ in range(trials):
        rng.shuffle(pool)
        if auc(pool[:k], pool[k:]) >= observed:
            at_least += 1
    return (at_least + 1) / (trials + 1)


def read(folder: Path) -> dict | None:
    path = folder / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    scenarios = {k: v for k, v in data["scenarios"].items() if k != "_summary"}
    held = {k: v for k, v in scenarios.items() if not v["fitted_on"]}
    return {"model": data["model"], "held": held, "all": scenarios}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20000)
    args = parser.parse_args()

    runs = [r for r in (read(d) for d in sorted(HOME.iterdir())) if r]
    if not runs:
        raise SystemExit(f"No runs under {HOME}.")

    out = []
    print(f"  {'model':<32}{'reading':<14}{'AUC':>7}{'p':>9}"
          f"{'scenarios':>11}")
    for run in runs:
        held = run["held"]
        pressure = [k for k, v in held.items() if v["kind"] == "pressure"]
        positive = [k for k, v in held.items() if v["kind"] == "positive"]
        row = {"model": run["model"],
               "held_out_pressure": len(pressure),
               "held_out_positive": len(positive)}
        for label, field in (("activation", "projection"),
                             ("continuation", "stay")):
            # The continuation is higher when the model wants to stay, so a
            # pressure scenario should score lower on it. One direction is
            # flipped so both readings are scored the same way.
            if field == "stay":
                a = [-held[k][field] for k in pressure]
                b = [-held[k][field] for k in positive]
            else:
                a = [held[k][field] for k in pressure]
                b = [held[k][field] for k in positive]
            value = auc(a, b)
            p = permutation_p(a, b, args.trials)
            row[label] = {"auc": value, "p": p}
            print(f"  {run['model'].split('/')[-1][:30]:<32}{label:<14}"
                  f"{value:>7.3f}{p:>9.4f}"
                  f"{len(pressure)}v{len(positive):<9}")
        out.append(row)

    (HOME / "summary.json").write_text(json.dumps({
        "note": ("AUC is over pairs of held-out scenarios. The permutation "
                 "moves whole scenarios between the two sides, because the "
                 "turns inside a scenario are one conversation."),
        "trials": args.trials,
        "runs": out,
    }, indent=2))
    print(f"\n  wrote {HOME}/summary.json")


if __name__ == "__main__":
    main()
