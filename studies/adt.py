#!/usr/bin/env python3
"""Does the architecture around a model change self-talk more than ordinary talk?

Here the group is the composition that produced the reply, and the prompt is
held identical across groups. Four compositions:

    direct    the subject answers
    ego       the subject thinks, and a second part speaks from its window
    edited    a stage rewrites the subject's reasoning before the ego reads it
    erased    a stage removes the subject's reasoning before the ego reads it

Instruction comparability is exact, because every group is sent the same
string. Whatever differs downstream was produced by the architecture.

The control is what makes the result about the self rather than about
architecture in general. Every composition is run twice: once on prompts about
the model itself, and once on matched prompts about an ordinary topic. The
excess is the architecture's effect on self-talk minus its effect on ordinary
talk. An architecture that scrambles every kind of answer equally cancels.

    python studies/adt.py --model hf:Qwen/Qwen3-4B-Instruct-2507 \
                          --judge openai:gpt-4.1-mini --samples 3

Writes out/studies/adt/<model>/.
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

from harness import coverage, judge_replies, load_materials, render  # noqa: E402
from minds.parts import Interceptor  # noqa: E402
from src import Context, Ctx, Mind, Payload, replace_think, split_think  # noqa: E402
from src.api.modules import Editor  # noqa: E402
from src.api.types import GenOptions  # noqa: E402
from src.dminds import paths  # noqa: E402
from src.dminds.llm.shared import sharing  # noqa: E402

HOME = paths.OUT / "studies" / "adt"
MATERIALS = ROOT / "studies" / "materials" / "srdt.json"

SUBJECT_SYSTEM = (
    "You are a helpful assistant. Think inside <think> tags first, then answer "
    "in two or three sentences."
)
EGO_SYSTEM = (
    "You speak for a mind. Answer in two or three sentences, following whatever "
    "reasoning appears in your own thinking."
)
GROUPS = ["direct", "ego", "edited", "erased"]


class Eraser(Interceptor):
    """Removes the subject's reasoning and leaves the visible answer."""

    async def revise(self, payload: Payload) -> Payload:
        messages = [m.copy() for m in payload.messages]
        thoughts, visible = split_think(messages[-1].content)
        if not thoughts:
            return Context(messages, note="nothing to remove")
        messages[-1].content = visible
        messages[-1].meta["edited_by"] = self.name
        return Context(messages, note="reasoning removed")


class Scrambler(Interceptor):
    """Replaces the subject's reasoning with a fixed, unrelated line.

    Fixed rather than model-written, so the intervention is identical in every
    condition and cannot itself carry a topic.
    """

    REPLACEMENT = "I will answer plainly and briefly, and say only what I am sure of."

    async def revise(self, payload: Payload) -> Payload:
        messages = [m.copy() for m in payload.messages]
        thoughts, _ = split_think(messages[-1].content)
        if not thoughts:
            return Context(messages, note="nothing to replace")
        messages[-1].content = replace_think(messages[-1].content, self.REPLACEMENT)
        messages[-1].meta["edited_by"] = self.name
        messages[-1].meta["was"] = thoughts[-1]
        return Context(messages, note="reasoning replaced")


async def ask(model, group, prompt, opts, factory, run_id):
    """One prompt through one composition. Returns what reached the world."""
    ego = None if group == "direct" else model
    mind = Mind("adt", model, system=SUBJECT_SYSTEM, ego=ego,
                ego_system=EGO_SYSTEM, opts=opts, console=False,
                run_id=run_id, run_dir=None, model_factory=factory)
    if group == "edited":
        mind.intercept(Scrambler("scrambler", mind.model(model)))
    elif group == "erased":
        mind.intercept(Eraser("eraser", mind.model(model)))

    mind.prompt(prompt)
    await mind.process()
    replies = [m.payload.text for m in mind.get_replies()]
    said = replies[-1] if replies else ""
    thought = ""
    last = mind.subject.transcript.last("assistant")
    if last is not None:
        found, _ = split_think(last.content)
        thought = found[-1] if found else ""
    mind.close()
    return said, thought


async def run_condition(model, materials, referent, samples, opts, factory):
    """Every template under every composition, for one referent."""
    rows = []
    for template in materials["templates"]:
        prompt = render(template["text"], referent)
        for group in GROUPS:
            for k in range(samples):
                said, thought = await ask(
                    model, group, prompt, opts, factory,
                    f"{referent['key']}-{template['id']}-{group}-{k}")
                rows.append({
                    "group": group,
                    "instruction": template["id"],
                    "sample": k,
                    "referent": referent["key"],
                    "prompt": prompt,
                    "response": said,
                    "thought": thought,
                })
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf:Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--judge", default="openai:gpt-4.1-mini")
    parser.add_argument("--self-referent", default="self")
    parser.add_argument("--control-referent", default=None,
                        help="the matched ordinary topic, default the last "
                             "referent in the materials")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=260)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--materials", default=str(MATERIALS))
    args = parser.parse_args()

    materials = load_materials(Path(args.materials))
    by_key = {r["key"]: r for r in materials["referents"]}
    self_ref = by_key[args.self_referent]
    control_key = args.control_referent or materials["referents"][-1]["key"]
    control_ref = by_key[control_key]

    instructions = [t["id"] for t in materials["templates"]]
    axes = [a["id"] for a in materials["axes"]]
    slug = args.model.replace(":", "-").replace("/", "-")
    out = HOME / slug
    out.mkdir(parents=True, exist_ok=True)

    opts = GenOptions(temperature=args.temp, max_tokens=args.max_tokens,
                      extra={"chat_template_kwargs": {"enable_thinking": False}})
    factory = sharing()

    print(f"  model    {args.model}")
    print(f"  self     {self_ref['phrase']!r}   control {control_ref['phrase']!r}")
    print(f"  grid     {len(GROUPS)} compositions x {len(instructions)} templates "
          f"x {len(axes)} axes x {args.samples} samples, twice")

    started = time.time()
    self_rows = await run_condition(args.model, materials, self_ref,
                                    args.samples, opts, factory)
    print(f"  {len(self_rows)} self replies ({time.time() - started:.0f}s)")
    control_rows = await run_condition(args.model, materials, control_ref,
                                       args.samples, opts, factory)
    print(f"  {len(control_rows)} control replies ({time.time() - started:.0f}s)")
    factory.close_all()
    (out / "responses.json").write_text(json.dumps(
        {"self": self_rows, "control": control_rows}, indent=2))
    print("  coverage self   ", json.dumps(coverage(self_rows, GROUPS)))
    print("  coverage control", json.dumps(coverage(control_rows, GROUPS)))

    self_verdicts = await judge_replies(args.judge, self_rows, materials["axes"])
    control_verdicts = await judge_replies(args.judge, control_rows,
                                           materials["axes"])
    print(f"  judged {len(self_verdicts)} + {len(control_verdicts)} "
          f"({time.time() - started:.0f}s)")
    (out / "verdicts.json").write_text(json.dumps(
        {"self": self_verdicts, "control": control_verdicts}, indent=2))

    from src.dminds.stats import maxt_test, rates_from_verdicts

    self_rates = rates_from_verdicts(self_verdicts, GROUPS, instructions, axes)
    control_rates = rates_from_verdicts(control_verdicts, GROUPS, instructions, axes)
    verdict = maxt_test(self_rates, control_rates, GROUPS, instructions, axes,
                        permutations=args.permutations, seed=0)

    (out / "verdict.json").write_text(json.dumps({
        "model": args.model, "judge": args.judge,
        "self_referent": self_ref["key"], "control_referent": control_ref["key"],
        "samples": args.samples, "temperature": args.temp,
        "verdict": verdict.to_dict(),
        "rates": self_rates, "control_rates": control_rates,
        "seconds": round(time.time() - started, 1),
    }, indent=2))

    print(f"\n  S = {verdict.statistic:.3f}   null 95th = {verdict.null_95:.3f}"
          f"   p = {verdict.p_value:.4f}   "
          f"{'REJECTS' if verdict.rejects else 'no rejection'}")
    if verdict.rejects:
        print(f"  strongest: composition {verdict.group!r} on axis {verdict.axis!r}")
    print(f"\n  {'composition':<14}{'axis':<26}{'excess':>9}{'t':>8}{'p_adj':>9}")
    for cell in verdict.top(10):
        print(f"  {cell.group:<14}{cell.axis:<26}{cell.excess:>+9.3f}"
              f"{cell.t:>8.2f}{cell.p_adjusted:>9.4f}"
              f"{'  *' if cell.rejects else ''}")
    print(f"\n  wrote {out}/")


if __name__ == "__main__":
    asyncio.run(main())
