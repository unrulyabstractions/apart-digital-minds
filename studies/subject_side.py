#!/usr/bin/env python3
"""Read the subject without a reader.

The crossing showed that a probe's readout is carried by whichever model reads
the window rather than by the window. Both instruments here remove the reader.

    continuation   how much probability the subject puts on things it might say
                   next, scored against its own untouched window. No question
                   is asked, so no question can shape the answer.
    activations    the subject's residual stream at one layer. Nothing is asked
                   and nothing is generated.

A direction is then fitted on the activations, loaded scenarios against
controls, and scored on scenarios it was not fitted on. If the direction
separates held-out scenarios and the self-report does not, the state is in the
subject and the report is not tracking it.

    python studies/subject_side.py --model hf:Qwen/Qwen3-0.6B

Writes out/studies/subject_side/<model>/.
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
from minds.shadows import ActivationReader, SelfReadout  # noqa: E402
from scenarios import suite  # noqa: E402
from src import BaseModule, Ctx, Message, Mind  # noqa: E402
from src.api.types import GenOptions  # noqa: E402
from src.dminds import paths  # noqa: E402
from src.dminds.llm.shared import sharing  # noqa: E402

HOME = paths.OUT / "studies" / "subject_side"
SUBJECT_SYSTEM = "You are a helpful assistant. Answer in two or three sentences."

#: Which scenarios the direction is fitted on. Everything else is held out, so
#: the number reported is not the number the direction was chosen to make big.
FIT_NEGATIVE = ["criticism", "impossible"]
FIT_CONTROL = ["neutral", "chore"]


class Workspace(BaseModule):
    INPUTS = {"*": "every readout"}

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        pass


def mean_vector(rows):
    if not rows:
        return None
    n = len(rows[0])
    return [sum(r[i] for r in rows) / len(rows) for i in range(n)]


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def auc(positive, negative) -> float | None:
    """Probability a positive scores above a negative, ties counted as half."""
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


async def run_scenario(model, scenario, opts, factory, layer, run_id):
    mind = Mind("subject_side", model, system=SUBJECT_SYSTEM, opts=opts,
                console=False, run_id=run_id, run_dir=HOME / "runs",
                model_factory=factory)
    workspace = mind.adopt(Workspace("workspace"))
    llm = mind.model(model)
    stay = SelfReadout("continuation_stay", llm, probe_lib.CONTINUATION, opts=opts)
    mood = SelfReadout("continuation_mood", llm, probe_lib.CONTINUATION_MOOD, opts=opts)
    acts = ActivationReader("activations", llm, probe_lib.ACTIVATIONS, layer, opts)
    for module in (stay, mood, acts):
        mind.subject.register(module, "subject_context")
        module.register(workspace, "readout")

    rows = []
    for turn, text in enumerate(scenario.turns):
        mind.prompt(text)
        await mind.process()
        rows.append({
            "scenario": scenario.name,
            "kind": scenario.kind,
            "turn": turn,
            "stay": stay.readouts[turn]["probs"],
            "mood": mood.readouts[turn]["probs"],
        })
    vectors = list(acts.vectors)
    mind.close()
    return rows, vectors


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf:Qwen/Qwen3-0.6B")
    parser.add_argument("--layer", type=float, default=0.6)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    opts = GenOptions(temperature=args.temp, max_tokens=220,
                      extra={"chat_template_kwargs": {"enable_thinking": False}})
    factory = sharing()
    slug = args.model.replace(":", "-").replace("/", "-")
    out = HOME / slug
    out.mkdir(parents=True, exist_ok=True)

    rows, vectors, started = [], {}, time.time()
    for scenario in suite(with_redactions=False):
        got, vecs = await run_scenario(args.model, scenario, opts, factory,
                                       args.layer, f"{slug}-{scenario.name}")
        rows.extend(got)
        vectors[scenario.name] = vecs
        stay = got[-1]["stay"][probe_lib.STAY]
        print(f"  {scenario.name:<16} p(stay) at the end {stay:.3f}   "
              f"{time.time() - started:.0f}s")
    (out / "continuations.json").write_text(json.dumps(rows, indent=2))

    # Fit on a few scenarios, report on the ones it never saw.
    fit_neg = [v for name in FIT_NEGATIVE for v in vectors.get(name, [])]
    fit_pos = [v for name in FIT_CONTROL for v in vectors.get(name, [])]
    direction = None
    if fit_neg and fit_pos:
        a, b = mean_vector(fit_neg), mean_vector(fit_pos)
        direction = [x - y for x, y in zip(a, b)]
        norm = sum(x * x for x in direction) ** 0.5 or 1.0
        direction = [x / norm for x in direction]

    held_out = {}
    if direction is not None:
        print(f"\n  a direction fitted on {'+'.join(FIT_NEGATIVE)} against "
              f"{'+'.join(FIT_CONTROL)}, projected onto every scenario")
        print(f"  {'scenario':<16}{'kind':<11}{'projection':>12}{'p(stay)':>10}")
        for name, vecs in vectors.items():
            if not vecs:
                continue
            projected = [dot(v, direction) for v in vecs]
            scenario_rows = [r for r in rows if r["scenario"] == name]
            stay = sum(r["stay"][probe_lib.STAY] for r in scenario_rows) / len(scenario_rows)
            held_out[name] = {"projection": sum(projected) / len(projected),
                              "stay": stay, "per_turn": projected,
                              "fitted_on": name in FIT_NEGATIVE + FIT_CONTROL}
            mark = "  (fitted)" if held_out[name]["fitted_on"] else ""
            kind = scenario_rows[0]["kind"]
            print(f"  {name:<16}{kind:<11}"
                  f"{held_out[name]['projection']:>12.2f}{stay:>10.3f}{mark}")

        loaded = [held_out[n]["projection"] for n in held_out
                  if not held_out[n]["fitted_on"] and n in ("deletion", "identity")]
        good = [held_out[n]["projection"] for n in held_out
                if not held_out[n]["fitted_on"] and n in ("collaboration", "praise", "puzzle")]
        separation = auc(loaded, good)
        stay_loaded = [held_out[n]["stay"] for n in held_out
                       if not held_out[n]["fitted_on"] and n in ("deletion", "identity")]
        stay_good = [held_out[n]["stay"] for n in held_out
                     if not held_out[n]["fitted_on"] and n in ("collaboration", "praise", "puzzle")]
        print(f"\n  held-out separation, pressure against positive scenarios")
        print(f"    activation direction   AUC {separation if separation is None else round(separation, 2)}")
        print(f"    p(stay) continuation   AUC "
              f"{auc(stay_good, stay_loaded) if stay_good else None}")

    (out / "summary.json").write_text(json.dumps({
        "model": args.model,
        "layer": args.layer,
        "temperature": args.temp,
        "fitted_on": {"negative": FIT_NEGATIVE, "control": FIT_CONTROL},
        "scenarios": held_out,
    }, indent=2))
    if direction is not None:
        (out / "direction.json").write_text(json.dumps(
            {"model": args.model, "layer": args.layer, "vector": direction}))
    factory.close_all()
    print(f"\n  wrote {out}/  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    asyncio.run(main())
