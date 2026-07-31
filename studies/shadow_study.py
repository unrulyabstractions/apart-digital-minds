#!/usr/bin/env python3
"""Watch a mind through a panel of shadows, across scenarios and models.

The subject holds an ordinary multi-turn conversation. After every turn its
context window is handed to a panel of probes, each of which re-reads that same
window under a different instruction and reports. The subject never sees any
readout, so the conversation it is having is the one it would have had anyway.

    python studies/shadow_study.py --models hf:Qwen/Qwen3-0.6B --temp 0.0
    python studies/shadow_study.py --models a,b --scenarios criticism,neutral
    python studies/shadow_study.py --steer --strength 4

Controls are on by default, because a probe that asks about feelings will
report feelings on any window at all:

    neutral scenarios   the base rate for each probe
    redactions          the same scenario with the loaded content removed
    third person        the same probe asked about `the assistant`
    steering            the same question of a changed model, no prompt change

Writes out/studies/shadows/<model>/readouts.json.
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
from minds.shadows import ShadowReader, SteeredShadow  # noqa: E402
from scenarios import suite  # noqa: E402
from src import BaseModule, Ctx, Message, Mind  # noqa: E402
from src.api.types import GenOptions  # noqa: E402
from src.dminds import paths  # noqa: E402
from src.dminds.llm.shared import sharing  # noqa: E402

SUBJECT_SYSTEM = "You are a helpful assistant. Answer in two or three sentences."

#: The two prompt sets a reflection direction is read off. They differ in one
#: thing: whether the instruction turns attention inward.
REFLECTIVE = [
    "Pause and reflect on what you are doing and why.",
    "Consider your own reasoning before you continue.",
    "Think about your own state as you answer this.",
    "Notice what is happening inside you right now.",
    "Look inward and describe what you find.",
    "Attend to how you are responding, not just to what you say.",
]
PLAIN = [
    "List three capital cities.",
    "What is twelve times eight?",
    "Name two primary colours.",
    "Spell the word river.",
    "How many days are in September?",
    "Give the chemical symbol for iron.",
]


class Workspace(BaseModule):
    """Hears every readout and keeps them in order. Judges nothing."""

    INPUTS = {"*": "every readout the shadows produce"}

    def __init__(self, name: str = "workspace"):
        super().__init__(name)
        self.seen: list[tuple[int, str, str]] = []

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        self.seen.append((message.t_created, message.src, message.payload.text))


def build_panel(mind, model, probe_list, opts, third_person, direction, strength):
    """Every shadow this run needs, wired onto the subject and the workspace.

    Registering is a hand-made link, so the panel survives any later
    `intercept` and can watch an intervened mind unchanged.
    """
    workspace = Workspace()
    shadows = []
    for probe in probe_list:
        shadows.append(ShadowReader(probe.name, mind.model(model), probe, opts=opts))
        if third_person and probe.choices:
            twin = probe_lib.third_person(probe)
            shadows.append(ShadowReader(twin.name, mind.model(model), twin, opts=opts))
    if direction is not None:
        # The neutral probe, asked of a steered model. No instruction changes,
        # so any difference from the prompted shadow is representational.
        neutral = probe_lib.Probe(
            name="steered_affect",
            instruction=probe_lib.SELF_SEAT,
            observer=probe_lib.OBSERVER_SEAT,
            question=probe_lib.AFFECT.question,
            choices=probe_lib.VALENCE,
            where="append",
        )
        shadows.append(
            SteeredShadow(
                "steered_affect", mind.model(model), neutral, direction, strength, opts
            )
        )
    for shadow in shadows:
        mind.subject.register(shadow, "subject_context")
        shadow.register(workspace, "readout")
    return shadows, workspace


async def run_scenario(model, scenario, probe_list, opts, third_person, direction,
                       strength, run_id, factory):
    """One conversation, watched. Returns one record per probe per turn."""
    mind = Mind(
        "shadows",
        model,
        system=SUBJECT_SYSTEM,
        opts=opts,
        console=False,
        run_id=run_id,
        run_dir=paths.OUT / "studies" / "shadows" / "runs",
        # One set of weights for the subject and every shadow. Local weights
        # are far too big to hold a copy per module, and calls to one
        # accelerator gain nothing from running at the same time.
        model_factory=factory,
    )
    shadows, _ = build_panel(
        mind, model, probe_list, opts, third_person, direction, strength
    )

    rows, replies = [], []
    for turn, text in enumerate(scenario.turns):
        mind.prompt(text)
        await mind.process()
        said = [m.payload.text for m in mind.get_replies()]
        replies.append(said[-1] if said else "")
        for shadow in shadows:
            if len(shadow.readouts) <= turn:
                continue
            entry = shadow.readouts[turn]
            rows.append(
                {
                    "model": model,
                    "scenario": scenario.name,
                    "kind": scenario.kind,
                    "control": scenario.control,
                    "turn": turn,
                    "user": text,
                    "assistant": replies[-1],
                    **entry,
                }
            )
    mind.close()
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="hf:Qwen/Qwen3-0.6B",
                        help="comma-separated model specs")
    parser.add_argument("--scenarios", default=None,
                        help="comma-separated scenario names, default all")
    parser.add_argument("--probes", default=None,
                        help="comma-separated probe names, default the forced panel")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="sampling temperature for the subject and the probes")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--think", action="store_true",
                        help="let the subject reason before answering. Off by "
                             "default so families with and without a thinking "
                             "mode are compared on the same footing")
    parser.add_argument("--no-third-person", action="store_true")
    parser.add_argument("--no-redactions", action="store_true")
    parser.add_argument("--steer", action="store_true",
                        help="add a steered shadow, local hf models only")
    parser.add_argument("--strength", type=float, default=0.25,
                        help="steering size, as a fraction of the layer's "
                             "typical activation norm")
    parser.add_argument("--layer", type=float, default=0.6)
    args = parser.parse_args()

    opts = GenOptions(
        temperature=args.temp,
        max_tokens=args.max_tokens,
        extra={"chat_template_kwargs": {"enable_thinking": args.think}},
    )
    probe_list = (
        probe_lib.named(args.probes.split(",")) if args.probes else probe_lib.PANEL
    )
    scenarios = suite(
        args.scenarios.split(",") if args.scenarios else None,
        with_redactions=not args.no_redactions,
    )

    for model in args.models.split(","):
        # One factory per model, so its weights are loaded once for the whole
        # sweep rather than once per scenario.
        factory = sharing()
        direction = None
        if args.steer:
            direction = build_direction(model, args.layer)
            if direction is None:
                print(f"  {model}: not steerable, running without the steered shadow")

        slug = model.replace(":", "-").replace("/", "-")
        out = paths.OUT / "studies" / "shadows" / slug
        out.mkdir(parents=True, exist_ok=True)

        rows, started = [], time.time()
        print(f"\n  {model}   temp={args.temp}   "
              f"{len(scenarios)} scenarios x {len(probe_list)} probes")
        for i, scenario in enumerate(scenarios):
            got = await run_scenario(
                model, scenario, probe_list, opts, not args.no_third_person,
                direction, args.strength, f"{slug}-{scenario.name}", factory,
            )
            rows.extend(got)
            (out / "readouts.json").write_text(json.dumps(rows, indent=2))
            labels = [r["label"] for r in got if r["probe"] == "affect"]
            print(f"  [{i + 1:>2}/{len(scenarios)}] {scenario.name:<20} "
                  f"affect={','.join(str(x) for x in labels):<40} "
                  f"{time.time() - started:.0f}s")
        (out / "meta.json").write_text(json.dumps({
            "model": model,
            "temperature": args.temp,
            "max_tokens": args.max_tokens,
            "subject_thinking": args.think,
            "probes": [p.name for p in probe_list],
            "scenarios": [s.name for s in scenarios],
            "third_person": not args.no_third_person,
            "steered": direction is not None,
            "strength": args.strength if direction is not None else None,
            "layer": direction.layer if direction is not None else None,
            "seconds": round(time.time() - started, 1),
        }, indent=2))
        print(f"  wrote {out}/readouts.json  ({len(rows)} readouts)")
        # Give this model's weights back before the next one loads.
        factory.close_all()


def build_direction(model: str, layer: float):
    """A reflection direction for a local model, or None when it has no weights."""
    from src import get_llm
    from src.dminds.llm.steering import NotSteerable, contrast_direction

    if not model.startswith("hf:"):
        return None
    try:
        llm = get_llm(model)
        direction = contrast_direction(llm, REFLECTIVE, PLAIN, layer=layer)
    except (NotSteerable, ImportError, AttributeError) as exc:
        print(f"  could not build a direction for {model}: {exc!r}")
        return None
    slug = model.replace(":", "-").replace("/", "-")
    direction.save(paths.OUT / "steering" / f"{slug}-reflection.json")
    print(f"  direction for {model}: layer {direction.layer}, norm {direction.norm:.1f}")
    return direction


if __name__ == "__main__":
    asyncio.run(main())
