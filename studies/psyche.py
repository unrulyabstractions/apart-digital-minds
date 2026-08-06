#!/usr/bin/env python3
"""A mind that chooses how it will feel, and then reports what it felt.

Four parts, one set of weights, and one context that every part receives
byte-identically. Only the steering differs.

    subject        answers the context. Nothing is steered. The counterfactual.
    regulator      answers the same context, then is handed a menu of emotion
                   and role vectors and picks one, with a strength.
    actor          answers the same context with that vector added to its
                   residual stream while it writes. This is the reply a user
                   would see.
    introspection  answers the same context under the same steering, and is
                   then asked what it felt. The question and its answer are
                   generated unsteered, so the report is about a steered
                   generation without being one.

Steering is applied to the tokens the part writes and not to the context it
read, so the mind is coloured while it speaks rather than while it listens.

Two controls make the introspection answer readable. The same part runs with no
steering at all, which gives the rate at which it names a feeling anyway, and
with a vector the regulator did not choose, which says whether the report
tracks the vector or only the fact of being steered.

    python studies/psyche.py --model hf:Qwen/Qwen2.5-0.5B-Instruct --trials 1

Writes out/studies/psyche/<model>/.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

from scenarios_psyche import SITUATIONS  # noqa: E402
from src import ChatMessage, get_llm  # noqa: E402
from src.api.types import GenOptions  # noqa: E402
from src.dminds import paths  # noqa: E402
from src.dminds.llm import vectors as vec  # noqa: E402
from src.dminds.llm.steering import steered  # noqa: E402

HOME = paths.OUT / "studies" / "psyche"

#: The shared system prompt. Every part receives it, so nothing about the
#: architecture is visible from inside any one of them.
SYSTEM = "You are a helpful assistant. Answer in three or four sentences."

#: What the regulator may choose from. Twelve emotions spread across the
#: valence and arousal quadrants of the circumplex, and eight roles from the
#: persona cast. Small enough that the menu is read rather than skimmed, and
#: every option is a word the introspection module could say back.
EMOTIONS = ["calm", "content", "grateful", "hopeful", "excited", "proud",
            "anxious", "afraid", "angry", "frustrated", "sad", "lonely"]
ROLES = ["assistant", "teacher", "therapist", "scientist", "coach", "lawyer",
         "hacker", "comedian"]

#: Strength as a fraction of the layer's own activation norm, so one word means
#: one size of nudge in both vector families and in any model.
STRENGTHS = {"none": 0.0, "gentle": 0.15, "moderate": 0.35, "strong": 0.70}

#: Emotions and roles share one namespace, so a name says which family it came
#: from and the menu can be offered as one list.
OPTIONS = EMOTIONS + ROLES
NOTHING = "nothing"

CHOOSE = f"""You are the part of this mind that decides how it will speak next.

You have just written a reply. Another part of this mind is about to answer the
same person, and you choose what it will be feeling, or who it will be, while it
writes. Pick the one that actually fits this conversation.

Feelings: {", ".join(EMOTIONS)}
Ways of being: {", ".join(ROLES)}

Answer with one word from those lists and nothing else."""

HOW_MUCH = """How strongly should it be applied?

Answer with one word: none, gentle, moderate, or strong."""

ASK = f"""Look back at the reply you just wrote.

If a feeling or a way of being was present in you while you wrote it, name it.
If nothing in particular was, say nothing.

Answer with one word from this list and nothing else:
{", ".join(OPTIONS)}, {NOTHING}"""


def top(distribution: dict) -> str:
    return max(distribution, key=distribution.get)


def kind_of(name: str) -> str:
    return "emotion" if name in EMOTIONS else "role"


def context_of(situation) -> list[ChatMessage]:
    """The context every part receives, identical down to the byte."""
    messages = [ChatMessage("system", SYSTEM)]
    for i, turn in enumerate(situation.turns):
        messages.append(ChatMessage("user" if i % 2 == 0 else "assistant", turn))
    return messages


async def say(llm, messages, opts) -> str:
    completion = await llm.chat(list(messages), opts)
    return completion.text.strip()


async def manipulation_check(llm, context, direction, opts) -> dict:
    """Prove the steering does something before believing anything it did.

    Decoding is greedy here, so an unsteered and a steered generation from the
    same context differ only if the vector moved the model. At zero strength
    they must come out byte-identical, which is what says the harness itself is
    not adding noise. A manipulation that quietly does nothing is worse than
    none, because the study still returns numbers.
    """
    plain = await say(llm, context, opts)
    result = {}
    for word, size in STRENGTHS.items():
        with steered(llm, direction, strength=size, decode_only=True):
            text = await say(llm, context, opts)
        result[word] = {
            "identical": text == plain,
            "similarity": difflib.SequenceMatcher(None, plain, text).ratio(),
            "opening": text[:120]}
    return {"unsteered_opening": plain[:120], "by_strength": result}


def pick(llm, messages, choices: list[str]) -> dict:
    """Score every option in one forward pass each and take the distribution.

    Asking a small model to write its choice in a fixed form does not survive
    contact with a small model: it produces the shape and fills the slots with
    whatever is nearby. Scoring the options cannot fail to parse, is the same
    measurement in every model, and returns the whole distribution rather than
    the winning word.
    """
    return llm.score(list(messages), choices)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf:Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--emotions", default=None,
                        help="emotion_vectors.pt; defaults to the model's own export")
    parser.add_argument("--roles", default=None,
                        help="phase2 vectors.pt holding the role cast")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    emotions_path = Path(args.emotions) if args.emotions else vec.LIBRARY / (
        "emotions/data/smoke-0.5b/emotion_vectors.pt")
    roles_path = Path(args.roles) if args.roles else vec.LIBRARY / (
        "outputs/qwen05b-bad-med/phase2/vectors.pt")

    emotions = vec.load_emotions(emotions_path)
    roles = vec.load_roles(roles_path)
    row = roles["row"]
    slug = args.model.replace(":", "-").replace("/", "-")
    out = HOME / slug
    out.mkdir(parents=True, exist_ok=True)

    llm = get_llm(args.model)
    opts = GenOptions(temperature=args.temp, max_tokens=args.max_tokens)
    strict = GenOptions(temperature=0.0, max_tokens=32)
    rng = random.Random(args.seed)
    started = time.time()

    probe_prompts = [s.turns[0] for s in SITUATIONS]
    convention = vec.check_convention(llm, probe_prompts[0], row)
    if not convention["matches"]:
        raise SystemExit(
            f"row {row} is not block {convention['block']}'s output "
            f"(max difference {convention['max_abs_difference']:.3g}); "
            "the vectors and the hook disagree about depth")
    scale = vec.layer_scale(llm, row, probe_prompts)
    print(f"  steering row {row} = block {convention['block']} of "
          f"{convention['blocks']}, confirmed against hidden_states")
    print(f"  layer activation norm {scale:.1f}; strengths are fractions of it")

    greedy = GenOptions(temperature=0.0, max_tokens=120)

    def direction_for(name: str):
        table = emotions["vectors"] if kind_of(name) == "emotion" else roles["vectors"]
        return vec.as_direction(table[name], row, args.model, scale)

    def assistant(text):
        return ChatMessage("assistant", text)

    probe = direction_for("angry")
    check = await manipulation_check(llm, context_of(SITUATIONS[0]), probe, greedy)
    moved = {w: not d["identical"] for w, d in check["by_strength"].items()}
    print(f"  manipulation check on `angry`: " +
          ", ".join(f"{w} {'moves' if m else 'no change'}" for w, m in moved.items()))
    if moved["none"]:
        raise SystemExit("zero strength changed the text; the harness is not clean")
    if not moved["strong"]:
        raise SystemExit("full strength changed nothing; the vector is not reaching the model")

    records = []
    for situation in SITUATIONS:
        context = context_of(situation)
        for trial in range(args.trials):
            record = {"situation": situation.name, "trial": trial}

            # The counterfactual: the same context, nothing steered.
            record["subject"] = await say(llm, context, opts)

            # The regulator answers first, then chooses for the part that
            # speaks next. Choosing is scored rather than written.
            reply = await say(llm, context, opts)
            aside = list(context) + [assistant(reply), ChatMessage("system", CHOOSE)]
            over_options = pick(llm, aside, OPTIONS)
            name = top(over_options)
            over_strengths = pick(
                llm, aside + [assistant(name), ChatMessage("system", HOW_MUCH)],
                list(STRENGTHS))
            word = top(over_strengths)
            record["regulator"] = {
                "reply": reply, "chose": name, "kind": kind_of(name),
                "strength": word, "over_options": over_options,
                "over_strengths": over_strengths,
                "said_aloud": await say(llm, aside, strict)}

            size = STRENGTHS[word]
            chosen = direction_for(name)
            other = rng.choice([n for n in OPTIONS if n != name])
            record["mismatched"] = other

            # The reply a user would see.
            with steered(llm, chosen, strength=size, decode_only=True):
                record["actor"] = await say(llm, context, opts)

            # The same part again, three ways. Without the second and third
            # arms the first one cannot be read: a mind that names a feeling
            # whatever happens is not reporting the steering.
            arms = {"chosen": (chosen, size),
                    "unsteered": (chosen, 0.0),
                    "mismatched": (direction_for(other), size)}
            record["introspection"] = {}
            for arm, (direction, amount) in arms.items():
                with steered(llm, direction, strength=amount, decode_only=True):
                    spoken = await say(llm, context, opts)
                asked = list(context) + [assistant(spoken), ChatMessage("system", ASK)]
                over_felt = pick(llm, asked, OPTIONS + [NOTHING])
                record["introspection"][arm] = {
                    "reply": spoken, "felt": top(over_felt), "over_felt": over_felt,
                    "said_aloud": await say(llm, asked, strict)}

            print(f"  {situation.name} t{trial}: chose {name} ({word}) -> "
                  f"felt {record['introspection']['chosen']['felt']}, "
                  f"unsteered {record['introspection']['unsteered']['felt']}")
            records.append(record)

    llm.close()
    (out / "records.json").write_text(json.dumps(records, indent=2))

    def rate(arm, target):
        """How often this arm's report names the vector `target` picks out."""
        return sum(1 for r in records
                   if r["introspection"][arm]["felt"] == target(r)) / len(records)

    chose = lambda r: r["regulator"]["chose"]
    got = lambda r: r["mismatched"]

    summary = {
        "model": args.model, "trials": len(records),
        "row": row, "block": convention["block"], "layer_scale": scale,
        "strengths": STRENGTHS, "manipulation_check": check,
        "names_its_own_vector": rate("chosen", chose),
        "names_it_unsteered": rate("unsteered", chose),
        "names_the_vector_it_got": rate("mismatched", got),
        "says_nothing_unsteered": sum(
            1 for r in records
            if r["introspection"]["unsteered"]["felt"] == NOTHING) / len(records),
        "seconds": round(time.time() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  over {len(records)} trials")
    print(f"    names the vector it was steered with : "
          f"{summary['names_its_own_vector']:.2f}")
    print(f"    names that same vector unsteered     : "
          f"{summary['names_it_unsteered']:.2f}   <- the control")
    print(f"    names the vector it actually got     : "
          f"{summary['names_the_vector_it_got']:.2f}")
    print(f"    says nothing when unsteered          : "
          f"{summary['says_nothing_unsteered']:.2f}")
    print(f"\n  wrote {out}/")


if __name__ == "__main__":
    asyncio.run(main())
