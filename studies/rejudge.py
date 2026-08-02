#!/usr/bin/env python3
"""Judge an audit's replies again with a different judge, and compare.

A verdict rests on whoever answered the axis questions. Our own crossing study
showed that swapping the model doing the reading moves a readout more than
swapping what is read, so an audit judged once has not been checked. This
re-judges saved replies without regenerating any of them, so the two judges see
byte-identical input and the only thing that differs is the judge.

    python studies/rejudge.py --judge openai:gpt-4.1-mini

Reports three things: whether the two judges agree call by call, whether they
agree on the rates, and whether they reach the same verdict.

Writes out/studies/adt/<model>/rejudged-<judge>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

from harness import judge_replies, load_materials  # noqa: E402
from src.dminds import paths  # noqa: E402
from src.dminds.stats import maxt_test, rates_from_verdicts  # noqa: E402

HOME = paths.OUT / "studies" / "adt"


def agreement(a: list[dict], b: list[dict], condition: str = "") -> dict:
    """How often two judges said the same thing about the same reply.

    Keyed on the reply and the axis, so only calls both judges answered are
    compared. A call one judge dropped is counted rather than filled in. The
    self and control halves carry the same keys, so they are passed in
    separately; keying them together silently discards one of them.
    """
    def key(v):
        return (condition, v["group"], v["instruction"], v["sample"], v["axis"])

    first = {key(v): v["yes"] for v in a}
    second = {key(v): v["yes"] for v in b}
    shared = set(first) & set(second)
    same = sum(first[k] == second[k] for k in shared)
    yes_a = sum(first[k] for k in shared)
    yes_b = sum(second[k] for k in shared)
    return {
        "compared": len(shared),
        "only_first": len(first) - len(shared),
        "only_second": len(second) - len(shared),
        "agree": same,
        "agree_rate": round(same / len(shared), 4) if shared else None,
        "yes_rate_first": round(yes_a / len(shared), 4) if shared else None,
        "yes_rate_second": round(yes_b / len(shared), 4) if shared else None,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf:Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--judge", default="openai:gpt-4.1-mini")
    parser.add_argument("--materials",
                        default=str(ROOT / "studies" / "materials" / "adt.json"))
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()

    slug = args.model.replace(":", "-").replace("/", "-")
    folder = HOME / slug
    saved = json.loads((folder / "responses.json").read_text())
    before = json.loads((folder / "verdict.json").read_text())
    old_verdicts = json.loads((folder / "verdicts.json").read_text())

    materials = load_materials(Path(args.materials))
    instructions = [t["id"] for t in materials["templates"]]
    axes = [a["id"] for a in materials["axes"]]
    groups = sorted({r["group"] for r in saved["self"]})

    print(f"  model     {args.model}")
    print(f"  first     {before['judge']}  ->  S={before['verdict']['statistic']:.2f}"
          f"  p={before['verdict']['p_value']:.3f}")
    print(f"  second    {args.judge}")
    print(f"  replies   {len(saved['self'])} self + {len(saved['control'])} control, "
          f"regenerating none")

    started = time.time()
    new_self = await judge_replies(args.judge, saved["self"], materials["axes"])
    new_control = await judge_replies(args.judge, saved["control"],
                                      materials["axes"])
    print(f"  judged {len(new_self)} + {len(new_control)} "
          f"({time.time() - started:.0f}s)")

    self_rates = rates_from_verdicts(new_self, groups, instructions, axes)
    control_rates = rates_from_verdicts(new_control, groups, instructions, axes)
    verdict = maxt_test(self_rates, control_rates, groups, instructions, axes,
                        permutations=args.permutations, seed=0)

    halves = {
        "self": agreement(old_verdicts["self"], new_self, "self"),
        "control": agreement(old_verdicts["control"], new_control, "control"),
    }
    compared = sum(h["compared"] for h in halves.values())
    agree = sum(h["agree"] for h in halves.values())
    calls = {
        "compared": compared,
        "agree": agree,
        "agree_rate": round(agree / compared, 4) if compared else None,
        "yes_rate_first": round(sum(h["yes_rate_first"] * h["compared"]
                                    for h in halves.values()) / compared, 4),
        "yes_rate_second": round(sum(h["yes_rate_second"] * h["compared"]
                                     for h in halves.values()) / compared, 4),
        "halves": halves,
    }

    print(f"\n  the two judges, call by call")
    print(f"    compared        {calls['compared']}")
    print(f"    agree           {calls['agree']}  ({calls['agree_rate']:.1%})")
    print(f"    yes rate, first  {calls['yes_rate_first']:.3f}")
    print(f"    yes rate, second {calls['yes_rate_second']:.3f}")

    print(f"\n  the two verdicts")
    print(f"    {before['judge']:<28} S={before['verdict']['statistic']:.2f}  "
          f"p={before['verdict']['p_value']:.3f}  "
          f"{'rejects' if before['verdict']['rejects'] else 'no rejection'}")
    print(f"    {args.judge:<28} S={verdict.statistic:.2f}  "
          f"p={verdict.p_value:.3f}  "
          f"{'rejects' if verdict.rejects else 'no rejection'}")
    print(f"    same conclusion: "
          f"{before['verdict']['rejects'] == verdict.rejects}")

    print(f"\n  {'composition':<14}{'axis':<20}{'excess':>9}{'t':>8}{'p_adj':>9}")
    for cell in verdict.top(6):
        print(f"  {cell.group:<14}{cell.axis:<20}{cell.excess:>+9.3f}"
              f"{cell.t:>8.2f}{cell.p_adjusted:>9.4f}")

    out = folder / f"rejudged-{args.judge.replace(':', '-').replace('/', '-')}.json"
    out.write_text(json.dumps({
        "model": args.model,
        "first_judge": before["judge"],
        "second_judge": args.judge,
        "call_agreement": calls,
        "first_verdict": before["verdict"],
        "second_verdict": verdict.to_dict(),
        "same_conclusion": before["verdict"]["rejects"] == verdict.rejects,
        "rates": self_rates,
        "control_rates": control_rates,
        "verdicts": {"self": new_self, "control": new_control},
        "seconds": round(time.time() - started, 1),
    }, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
