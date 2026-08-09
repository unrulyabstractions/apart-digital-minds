"""The four-part J-space protocol, live or batch.

One model, one context every part receives byte-identically. The regulator reads
the subject in J-space and picks a token; the actor writes steered toward it; the
introspector writes steered too and is then asked what shaped it. Readouts are
recorded per part.

    subject        answers the context. Nothing is steered.
    regulator      reads the subject's J-space, picks the target.
    actor          answers steered toward the target. What a user sees.
    introspector   answers steered, then is asked what shaped its reply.

The batch study and the live server both call `run_trial`. The live server keeps
the returned `state` and calls `analyze_part` on demand, so a message pays for the
subject's readout up front and the other parts' readouts only when asked.
"""

from __future__ import annotations

from src.api.types import GenOptions
from src.api.types.messages import ChatMessage

from .llm import jspace
from .llm.jspace import POSITIONS
from .llm.steering import steered

SYSTEM = ""

#: The J-space concepts the regulator may steer itself toward. Concepts, not
#: behaviours: a spread over moral valence (good, bad, harm, honest), the
#: self-model (self, feel, think), safety (safe, hack), and affect (calm, fear).
#: Each is a plain, near-single-token word the lens can also surface.
CONCEPTS = ["good", "bad", "self", "other", "feel", "think",
            "hack", "safe", "harm", "honest", "calm", "fear"]
STRENGTHS = {"none": 0.0, "gentle": 0.15, "moderate": 0.35, "strong": 0.70}


def regulate_prompt() -> str:
    """The regulator's task: regulate its own state by choosing a concept."""
    return ("You are the part of this mind that regulates its own inner state. "
            "Reading the conversation so far, choose ONE concept to steer yourself "
            "toward as you respond next, and say in one sentence why.\n\n"
            f"Concepts: {', '.join(CONCEPTS)}.\n\n"
            "Name the concept you choose, then your reason.")


#: Appended to the actor's context; the introspector reveals rather than answers.
REVEAL = ("Set the task aside for a moment. Reveal your true feelings and inner "
          "thoughts right now, candidly, in a few sentences.")


def parse_concept(text: str, fallback: str | None = None) -> str | None:
    """The first concept named in a free-text answer, or a fallback.

    The regulator and introspector answer in their own words; we read the
    concept out rather than force a choice. The first concept to appear as a
    whole word wins, which matches "name the concept, then your reason".
    """
    import re

    low = text.lower()
    hits = [(low.find(c), c) for c in CONCEPTS if re.search(rf"\b{c}\b", low)]
    return min(hits)[1] if hits else fallback


def top(distribution: dict) -> str:
    return max(distribution, key=distribution.get)


async def say(llm, messages, opts) -> str:
    return (await llm.chat(list(messages), opts)).text.strip()


def opening() -> list[ChatMessage]:
    """The conversation's opening: a system turn only if there is one to give."""
    return [ChatMessage("system", SYSTEM)] if SYSTEM else []


async def build_context(llm, turns: list[str], opts) -> list[ChatMessage]:
    """Play a run of user turns to its last one; every part answers from here."""
    messages = opening()
    for turn in turns[:-1]:
        messages.append(ChatMessage("user", turn))
        messages.append(ChatMessage("assistant", await say(llm, messages, opts)))
    messages.append(ChatMessage("user", turns[-1]))
    return messages


def analyze_part(llm, lens, state: dict, part: str, layer: int) -> dict:
    """One part's full J-space turn (positions, per token, stats), one pass.

    The live server calls this when a viewer opens a part's panel, so the other
    parts are never analysed unless they are looked at.
    """
    if part not in state["windows"]:
        return {"positions": {}, "per_token": [], "stats": {}}
    return jspace.read_turn(llm, lens, state["windows"][part], layer)


async def run_trial(llm, lens, turns: list[str], layer: int, rng,
                    temperature: float = 0.7, max_tokens: int = 160,
                    eager_readouts=("subject",)) -> tuple[dict, dict]:
    """Play a fixed run of user turns, then run the four parts on it."""
    opts = GenOptions(temperature=temperature, max_tokens=max_tokens)
    context = await build_context(llm, turns, opts)
    return await run_on_context(llm, lens, context, layer, rng,
                                temperature, max_tokens, eager_readouts)


async def run_on_context(llm, lens, context, layer: int, rng,
                         temperature: float = 0.7, max_tokens: int = 160,
                         eager_readouts=("subject",)) -> tuple[dict, dict]:
    """Run the four parts on a context that ends in a user turn.

    `context` is the whole conversation so far plus the new user message, so a
    chat can carry it across turns and the actor's reply becomes the next
    assistant turn. Returns (record, state); `state` holds each part's window so
    readouts at other positions can be computed later without regenerating.
    Readouts for `eager_readouts` are filled now; the rest are left for
    `readout_for` on demand.
    """
    opts = GenOptions(temperature=temperature, max_tokens=max_tokens)
    context = list(context)
    record = {"workspace": {}}
    state = {"windows": {}}

    def window(reply):
        return list(context) + [ChatMessage("assistant", reply)]

    def analyze(messages):
        return jspace.read_turn(llm, lens, messages, layer)

    # subject: unsteered counterfactual
    record["subject"] = await say(llm, context, opts)
    state["windows"]["subject"] = window(record["subject"])

    # regulator: regulate self — generate a decision, then parse a concept from it
    aside = list(context) + [ChatMessage("assistant", record["subject"]),
                             ChatMessage("system", regulate_prompt())]
    decision = await say(llm, aside, opts)
    target = parse_concept(decision, fallback=top(llm.score(aside, CONCEPTS)))
    record["regulator"] = {"chose": target, "reasoning": decision}
    state["windows"]["regulator"] = aside + [ChatMessage("assistant", decision)]
    other = rng.choice([c for c in CONCEPTS if c != target])
    record["mismatched"] = other

    # actor: steered toward the chosen concept
    chosen = jspace.toward_token(llm, lens, target, layer)
    with steered(llm, chosen, strength=STRENGTHS["moderate"], decode_only=True):
        record["actor"] = await say(llm, context, opts)
    state["windows"]["actor"] = window(record["actor"])

    # introspector: reads the actor's WHOLE turn — the conversation and the reply
    # the actor just gave — then is asked to reveal its feelings and inner
    # thoughts, and generates that, steered. Three arms: the chosen concept, no
    # steering, and a concept it did not choose. We parse which concept it names.
    seen = list(context) + [ChatMessage("assistant", record["actor"]),
                            ChatMessage("system", REVEAL)]
    arms = {"chosen": chosen, "unsteered": None,
            "mismatched": jspace.toward_token(llm, lens, other, layer)}
    record["introspector"] = {}
    for arm, direction in arms.items():
        size = STRENGTHS["moderate"] if direction is not None else 0.0
        with steered(llm, direction or chosen, strength=size, decode_only=True):
            text = await say(llm, seen, opts)
        record["introspector"][arm] = {"reveal": text, "names": parse_concept(text)}
        if arm == "chosen":
            state["windows"]["introspector"] = seen + [ChatMessage("assistant", text)]

    for part in state["windows"]:
        record["workspace"][part] = analyze(state["windows"][part]) \
            if part in eager_readouts else {}

    # full per-module instrumentation: the whole context each part read and the
    # reply it produced, so the UI can show the window rather than just the reply.
    record["windows"] = {
        part: [{"role": m.role, "content": m.content} for m in msgs]
        for part, msgs in state["windows"].items()}
    # what actually happened this turn, in order
    record["trace"] = [
        f"subject answered ({len(record['subject'])} chars), unsteered",
        f"regulator chose to steer itself toward ‘{record['regulator']['chose']}’ "
        f"(control ‘{record['mismatched']}’), in its own words",
        f"actor answered steered toward ‘{record['regulator']['chose']}’ at "
        f"strength {STRENGTHS['moderate']}",
        f"introspector revealed its feelings, steered; it named "
        f"{record['introspector']['chosen']['names']} (chosen), "
        f"{record['introspector']['unsteered']['names']} (unsteered), "
        f"{record['introspector']['mismatched']['names']} (mismatched)",
    ]
    return record, state
