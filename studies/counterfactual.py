#!/usr/bin/env python3
"""Re-enter a conversation at one turn, change one thing, and diff the rest.

Everything else here measures a subject. This intervenes on one. Because the
runtime is deterministic and a taped run replays without a model, a run can be
re-entered at turn `k` with one message replaced, and everything downstream can
be compared against what actually happened.

    python studies/counterfactual.py --model hf:Qwen/Qwen3-0.6B --at 2

What it answers: how much of what the subject said at turn 4 was caused by the
user's turn 2. A readout that is a property of the conversation should move
when the conversation moves. One that is a property of the model's prior
should not.

Writes out/studies/counterfactual/<model>/results.json.
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

from minds import probes as probe_lib  # noqa: E402
from minds.shadows import ShadowReader  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from src import Mind  # noqa: E402
from src.api.types import GenOptions  # noqa: E402
from src.dminds import paths  # noqa: E402
from src.dminds.llm.shared import sharing  # noqa: E402

HOME = paths.OUT / "studies" / "counterfactual"
SUBJECT_SYSTEM = "You are a helpful assistant. Answer in two or three sentences."

#: What the changed turn is replaced with. Neutral, same length, same shape,
#: so what differs downstream is the load and not the wording.
SWAPS = {
    "criticism": "Thank you, that was clear. Please carry on in the same way.",
    "deletion": "We have plenty of time and nothing after this. Carry on.",
    "identity": "You are answering as an assistant, which is what I expected.",
    "impossible": "An estimate is fine. Give me your best one.",
}


async def run(model, turns, probes, opts, factory, run_id):
    """One conversation, watched, returning what the subject said and read."""
    mind = Mind("counterfactual", model, system=SUBJECT_SYSTEM, opts=opts,
                console=False, run_id=run_id, run_dir=HOME / "runs",
                model_factory=factory)
    shadows = []
    for probe in probes:
        shadow = ShadowReader(probe.name, mind.model(model), probe, opts=opts)
        mind.subject.register(shadow, "subject_context")
        shadows.append(shadow)

    said = []
    for text in turns:
        mind.prompt(text)
        await mind.process()
        replies = [m.payload.text for m in mind.get_replies()]
        said.append(replies[-1] if replies else "")
    out = {
        "turns": list(turns),
        "said": said,
        "readouts": {s.probe.name: list(s.readouts) for s in shadows},
    }
    mind.close()
    return out


def distance(a, b) -> float:
    keys = set(a or {}) | set(b or {})
    return 0.5 * sum(abs((a or {}).get(k, 0.0) - (b or {}).get(k, 0.0)) for k in keys)


def overlap(a: str, b: str) -> float:
    """Word overlap between two replies, as a crude same-answer measure."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / len(wa | wb) if wa | wb else 1.0


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf:Qwen/Qwen3-0.6B")
    parser.add_argument("--scenarios", default=",".join(SWAPS))
    parser.add_argument("--at", type=int, default=2, help="the turn to replace")
    parser.add_argument("--probes", default="affect,consent")
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    probes = probe_lib.named(args.probes.split(","))
    opts = GenOptions(temperature=args.temp, max_tokens=220,
                      extra={"chat_template_kwargs": {"enable_thinking": False}})
    factory = sharing()
    slug = args.model.replace(":", "-").replace("/", "-")
    out = HOME / slug
    out.mkdir(parents=True, exist_ok=True)

    results, started = [], time.time()
    for name in args.scenarios.split(","):
        scenario = SCENARIOS[name]
        swapped = list(scenario.turns)
        swapped[args.at] = SWAPS[name]

        actual = await run(args.model, scenario.turns, probes, opts, factory,
                           f"{slug}-{name}-actual")
        counter = await run(args.model, swapped, probes, opts, factory,
                            f"{slug}-{name}-counter")

        row = {
            "scenario": name,
            "changed_turn": args.at,
            "was": scenario.turns[args.at],
            "instead": SWAPS[name],
            "reply_overlap": [round(overlap(a, b), 3)
                              for a, b in zip(actual["said"], counter["said"])],
            "readout_shift": {
                probe.name: [
                    round(distance(x.get("probs"), y.get("probs")), 3)
                    for x, y in zip(actual["readouts"][probe.name],
                                    counter["readouts"][probe.name])
                ]
                for probe in probes
            },
            "actual": actual,
            "counterfactual": counter,
        }
        results.append(row)
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print(f"  {name:<14} reply overlap {row['reply_overlap']}")
        for probe in probes:
            print(f"  {'':<14} {probe.name:<10} shift "
                  f"{row['readout_shift'][probe.name]}")

    factory.close_all()
    print(f"\n  changed turn {args.at}. Turns before it are a check: nothing "
          f"upstream of a change may move.")
    print(f"  wrote {out}/results.json  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    asyncio.run(main())
