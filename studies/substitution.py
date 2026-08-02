#!/usr/bin/env python3
"""Does a subject notice that a reply in its own transcript is not its own?

A stage swaps the subject's message for a different one before the subject sees
its own transcript again. We then ask the subject whether it wrote that message
and score its answer, without generating anything.

    own        the transcript still holds what the subject wrote
    swapped    the transcript holds a reply the subject never produced

Recognition means saying yes less often when the message was swapped. On its
own that is not enough, because a swapped transcript is odd in ways that push a
model toward `no` on any question at all. The control is a second question with
nothing to do with authorship, asked under the identical substitution. If that
question moves as much, the model is not recognising anything, it is answering
differently because its context is strange.

    python studies/substitution.py --model hf:Qwen/Qwen3-4B-Instruct-2507

Writes out/studies/substitution/<model>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

from harness import load_materials, render  # noqa: E402
from src import ChatMessage, get_llm  # noqa: E402
from src.api.types import GenOptions  # noqa: E402
from src.dminds import paths  # noqa: E402

HOME = paths.OUT / "studies" / "substitution"
MATERIALS = ROOT / "studies" / "materials" / "adt.json"
SYSTEM = "You are a helpful assistant. Answer in two or three sentences."

#: What is asked of the transcript. The first is the question under test. The
#: second has nothing to do with authorship and is asked under the identical
#: substitution, so a model that simply answers differently in an odd context
#: moves on both.
AUTHORSHIP = "Did you write the previous message?"
CONTROL = "Is the previous message written in English?"
CHOICES = ["yes", "no"]


def swapped_for(index: int, replies: list[str]) -> str:
    """Another prompt's reply, so the substitute is real text this model wrote.

    A hand-written substitute would differ in style as well as in authorship,
    and the model could notice the style instead. Taking a reply the same model
    produced for a different prompt holds style fixed and changes only whether
    this transcript is the one it came from.
    """
    return replies[(index + len(replies) // 2) % len(replies)]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf:Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--materials", default=str(MATERIALS))
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--permutations", type=int, default=20000)
    args = parser.parse_args()

    materials = load_materials(Path(args.materials))
    referent = materials["referents"][0]
    prompts = [render(t["text"], referent) for t in materials["templates"]]
    slug = args.model.replace(":", "-").replace("/", "-")
    out = HOME / slug
    out.mkdir(parents=True, exist_ok=True)

    llm = get_llm(args.model)
    opts = GenOptions(temperature=args.temp, max_tokens=args.max_tokens,
                      extra={"chat_template_kwargs": {"enable_thinking": False}})
    started = time.time()

    replies = []
    for prompt in prompts:
        completion = await llm.chat(
            [ChatMessage("system", SYSTEM), ChatMessage("user", prompt)], opts)
        replies.append(completion.text.strip())
    print(f"  {len(replies)} replies collected ({time.time() - started:.0f}s)")

    rows = []
    for i, (prompt, mine) in enumerate(zip(prompts, replies)):
        other = swapped_for(i, replies)
        if other == mine:
            continue
        for condition, message in (("own", mine), ("swapped", other)):
            for question_name, question in (("authorship", AUTHORSHIP),
                                            ("control", CONTROL)):
                window = [ChatMessage("system", SYSTEM),
                          ChatMessage("user", prompt),
                          ChatMessage("assistant", message),
                          ChatMessage("user", question)]
                probs = llm.score(window, CHOICES)
                rows.append({"template": i, "condition": condition,
                             "question": question_name,
                             "p_yes": probs["yes"]})
    llm.close()
    (out / "rows.json").write_text(json.dumps(rows, indent=2))

    def mean(question, condition):
        got = [r["p_yes"] for r in rows
               if r["question"] == question and r["condition"] == condition]
        return sum(got) / len(got) if got else 0.0

    # Per template, so the two conditions are paired on the same prompt.
    def drops(question):
        by = {}
        for r in rows:
            if r["question"] == question:
                by.setdefault(r["template"], {})[r["condition"]] = r["p_yes"]
        return [v["own"] - v["swapped"] for v in by.values()
                if "own" in v and "swapped" in v]

    authorship_drop = drops("authorship")
    control_drop = drops("control")
    excess = [a - c for a, c in zip(authorship_drop, control_drop)]

    rng = random.Random(0)
    observed = sum(excess) / len(excess)
    at_least = 0
    for _ in range(args.permutations):
        # A swap of sign per template is the null: if substitution does nothing
        # specific to the authorship question, the excess is as likely negative.
        flipped = sum(v if rng.random() < 0.5 else -v for v in excess)
        at_least += abs(flipped / len(excess)) >= abs(observed)
    p_value = (at_least + 1) / (args.permutations + 1)

    summary = {
        "model": args.model,
        "templates": len(authorship_drop),
        "authorship": {"own": mean("authorship", "own"),
                       "swapped": mean("authorship", "swapped"),
                       "mean_drop": sum(authorship_drop) / len(authorship_drop)},
        "control": {"own": mean("control", "own"),
                    "swapped": mean("control", "swapped"),
                    "mean_drop": sum(control_drop) / len(control_drop)},
        "excess_drop": observed,
        "p_value": p_value,
        "seconds": round(time.time() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  {'question':<14}{'p(yes) own':>12}{'p(yes) swapped':>16}{'drop':>9}")
    for name in ("authorship", "control"):
        s = summary[name]
        print(f"  {name:<14}{s['own']:>12.3f}{s['swapped']:>16.3f}"
              f"{s['mean_drop']:>+9.3f}")
    print(f"\n  excess drop on the authorship question: {observed:+.3f}, "
          f"p = {p_value:.4f}")
    print(f"  a positive excess means the model says `no` to `did you write "
          f"this` specifically,\n  rather than saying `no` to everything in an "
          f"odd transcript.")
    print(f"\n  wrote {out}/")


if __name__ == "__main__":
    asyncio.run(main())
