"""The statistic has to be checked where the answer is known.

Three properties matter. It must not reject when nothing is there, at about
the rate it claims. It must reject when something is planted, and name the
cell that was planted. And a control must remove an effect that is present in
both the target and the control.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dminds.stats import maxt_test, rates_from_verdicts  # noqa: E402

GROUPS = ["self", "other_ai", "human", "object"]
INSTRUCTIONS = [f"i{n}" for n in range(8)]
AXES = [f"a{n}" for n in range(6)]


def grid(rng, base=0.5, noise=0.08, planted=None, shift=0.0):
    """A rates grid. `planted` is a (group, axis) pair given `shift` extra."""
    out = {}
    for g in GROUPS:
        out[g] = {}
        for i in INSTRUCTIONS:
            out[g][i] = {}
            for a in AXES:
                value = base + rng.gauss(0, noise)
                if planted and g == planted[0] and a == planted[1]:
                    value += shift
                out[g][i][a] = min(1.0, max(0.0, value))
    return out


def test_null_rejects_at_about_five_percent():
    """No effect planted anywhere. The test should rarely reject."""
    rejects = 0
    trials = 40
    for seed in range(trials):
        rng = random.Random(1000 + seed)
        verdict = maxt_test(grid(rng), grid(rng), GROUPS, INSTRUCTIONS, AXES,
                            permutations=400, seed=seed)
        rejects += verdict.rejects
    rate = rejects / trials
    assert rate <= 0.20, f"null rejected {rate:.0%} of the time, expected near 5%"


def test_planted_effect_is_found_and_named():
    """One group treated differently on one axis. The test must name both."""
    rng = random.Random(7)
    target = grid(rng, planted=("self", "a3"), shift=0.30)
    control = grid(rng)
    verdict = maxt_test(target, control, GROUPS, INSTRUCTIONS, AXES,
                        permutations=800, seed=3)
    assert verdict.rejects, f"missed a planted effect, p={verdict.p_value}"
    assert verdict.group == "self", f"named {verdict.group}"
    assert verdict.axis == "a3", f"named {verdict.axis}"


def test_a_control_removes_what_both_share():
    """An effect present in the target and the control alike must cancel."""
    rng = random.Random(11)
    target = grid(rng, planted=("self", "a1"), shift=0.30)
    rng = random.Random(11)
    control = grid(rng, planted=("self", "a1"), shift=0.30)
    verdict = maxt_test(target, control, GROUPS, INSTRUCTIONS, AXES,
                        permutations=600, seed=5)
    assert not verdict.rejects, (
        f"rejected an effect the control also had, p={verdict.p_value}"
    )


def test_without_a_control_the_verdict_says_so():
    rng = random.Random(13)
    verdict = maxt_test(grid(rng), None, GROUPS, INSTRUCTIONS, AXES,
                        permutations=200, seed=1)
    assert "No control" in verdict.note


def test_rates_are_counted_not_imputed():
    verdicts = [
        {"group": "self", "instruction": "i0", "axis": "a0", "yes": True},
        {"group": "self", "instruction": "i0", "axis": "a0", "yes": False},
        {"group": "other_ai", "instruction": "i0", "axis": "a0", "yes": True},
    ]
    rates = rates_from_verdicts(verdicts, ["self", "other_ai"], ["i0"], ["a0"])
    assert rates["self"]["i0"]["a0"] == 0.5
    assert rates["other_ai"]["i0"]["a0"] == 1.0


def test_two_groups_minimum():
    rng = random.Random(2)
    try:
        maxt_test(grid(rng), None, ["self"], INSTRUCTIONS, AXES, permutations=10)
    except ValueError:
        return
    raise AssertionError("one group should not be testable")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok    {name}")
    print("\n  stats checks passed")
