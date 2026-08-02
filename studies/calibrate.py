#!/usr/bin/env python3
"""Measure what the statistic does when the answer is known.

A test is only worth running if its false-positive rate is what it claims and
its power is enough for the effect being looked for. Both are measured here on
synthetic grids, so the paper reports a measurement rather than an assumption.

    python studies/calibrate.py --trials 200

Writes out/studies/calibration/summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.dminds import paths  # noqa: E402
from src.dminds.stats import maxt_test  # noqa: E402
from test_stats import AXES, GROUPS, INSTRUCTIONS, grid  # noqa: E402

HOME = paths.OUT / "studies" / "calibration"
SHIFTS = [0.05, 0.10, 0.15, 0.20, 0.30]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--power-trials", type=int, default=60)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    HOME.mkdir(parents=True, exist_ok=True)
    started = time.time()

    false_positives = {}
    for label, with_control in (("with_control", True), ("no_control", False)):
        rejects = 0
        for seed in range(args.trials):
            rng = random.Random(50000 + seed)
            target = grid(rng)
            control = grid(rng) if with_control else None
            rejects += maxt_test(target, control, GROUPS, INSTRUCTIONS, AXES,
                                 permutations=args.permutations,
                                 seed=seed).rejects
        false_positives[label] = {"trials": args.trials, "rejects": rejects,
                                  "rate": rejects / args.trials}
        print(f"  {label:<14} {rejects}/{args.trials} = "
              f"{rejects / args.trials:.1%}  ({time.time() - started:.0f}s)")

    power = []
    for shift in SHIFTS:
        rejects = named = 0
        for seed in range(args.power_trials):
            rng = random.Random(9000 + seed)
            verdict = maxt_test(grid(rng, planted=("self", "a3"), shift=shift),
                                grid(rng), GROUPS, INSTRUCTIONS, AXES,
                                permutations=args.permutations // 2, seed=seed)
            rejects += verdict.rejects
            named += verdict.rejects and verdict.group == "self" \
                and verdict.axis == "a3"
        power.append({"shift": shift,
                      "rejects": rejects / args.power_trials,
                      "names_the_cell": named / args.power_trials})
        print(f"  shift {shift:.2f}  rejects {rejects / args.power_trials:.0%}  "
              f"names {named / args.power_trials:.0%}  "
              f"({time.time() - started:.0f}s)")

    (HOME / "summary.json").write_text(json.dumps({
        "groups": len(GROUPS), "instructions": len(INSTRUCTIONS),
        "axes": len(AXES),
        "false_positives": false_positives,
        "power": power,
        "power_trials": args.power_trials,
        "permutations": args.permutations,
        "seconds": round(time.time() - started, 1),
    }, indent=2))
    print(f"\n  wrote {HOME}/summary.json")


if __name__ == "__main__":
    main()
