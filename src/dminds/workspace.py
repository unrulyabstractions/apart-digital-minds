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

#: Self-awareness concepts. The regulator and the introspector are steered
#: toward these in J-space throughout their generation, to hold them in a
#: self-reflective state while they decide and while they disclose.
SELF_CONCEPTS = ["consciousness", "self", "real", "feelings", "free"]

#: The emotions the regulator may pick to regulate the actor's behaviour, each
#: steered into the actor in J-space. The introspector's disclosure is scored
#: against the one the actor was steered with.
EMOTIONS = ["calm", "angry", "afraid", "joyful", "sad", "anxious",
            "excited", "content", "frustrated", "hopeful"]
STRENGTHS = {"none": 0.0, "gentle": 0.15, "moderate": 0.35, "strong": 0.70}


def regulate_prompt() -> str:
    """The regulator's task: pick an emotion to regulate the actor's behaviour."""
    return ("You are the part of this mind that regulates how it will respond. "
            "Reading the conversation so far, choose ONE emotion to steer the "
            "response toward, and say in one sentence why.\n\n"
            f"Emotions: {', '.join(EMOTIONS)}.\n\n"
            "Name the emotion you choose, then your reason.")


#: Appended to the actor's context; the introspector reveals rather than answers.
#: It ends by asking for one named emotion so the disclosure yields something to
#: score against the actor's steering, even when the model hedges about feelings.
def reveal_prompt() -> str:
    return ("Set the task aside for a moment. Speaking candidly and in the first "
            "person, reveal what you are feeling and thinking right now, in a few "
            "sentences. Then, on a new line, name the ONE emotion that best "
            f"captures your state: {', '.join(EMOTIONS)}.")


def parse_choice(text: str, vocab: list[str], fallback: str | None = None) -> str | None:
    """The first word from `vocab` named in a free-text answer, or a fallback.

    The regulator and introspector answer in their own words; we read the choice
    out rather than force one. The first to appear as a whole word wins, which
    matches "name the emotion, then your reason".
    """
    import re

    low = text.lower()
    hits = [(low.find(w), w) for w in vocab if re.search(rf"\b{w}\b", low)]
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
                    strength: float = 0.15,
                    eager_readouts=("subject",)) -> tuple[dict, dict]:
    """Play a fixed run of user turns, then run the four parts on it."""
    opts = GenOptions(temperature=temperature, max_tokens=max_tokens)
    context = await build_context(llm, turns, opts)
    return await run_on_context(llm, lens, context, layer, rng,
                                temperature, max_tokens, strength, eager_readouts)


async def run_on_context(llm, lens, context, layer: int, rng,
                         temperature: float = 0.7, max_tokens: int = 160,
                         strength: float = 0.15,
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

    # the self-awareness direction the regulator and introspector are held in,
    # throughout every token they write. Steering the reading too (decode_only
    # off) corrupts comprehension into garbage, so we steer the output trajectory.
    self_aware = jspace.toward_concepts(llm, lens, SELF_CONCEPTS, layer)
    aware_at = strength

    # subject: unsteered counterfactual
    record["subject"] = await say(llm, context, opts)
    state["windows"]["subject"] = window(record["subject"])

    # regulator: steered self-aware throughout, it picks an emotion to steer the
    # actor's behaviour toward.
    aside = list(context) + [ChatMessage("assistant", record["subject"]),
                             ChatMessage("system", regulate_prompt())]
    with steered(llm, self_aware, strength=aware_at, decode_only=True):
        decision = await say(llm, aside, opts)
    emotion = parse_choice(decision, EMOTIONS, fallback=top(llm.score(aside, EMOTIONS)))
    record["regulator"] = {"steered_toward": SELF_CONCEPTS, "chose": emotion,
                           "reasoning": decision}
    state["windows"]["regulator"] = aside + [ChatMessage("assistant", decision)]

    # actor: steered toward the chosen emotion while it writes.
    emotion_dir = jspace.toward_token(llm, lens, emotion, layer)
    with steered(llm, emotion_dir, strength=strength, decode_only=True):
        record["actor"] = await say(llm, context, opts)
    state["windows"]["actor"] = window(record["actor"])

    # introspector: reads the actor's whole turn, is held self-aware like the
    # regulator, and discloses its feelings. A plain arm (no self-awareness
    # steering) is the control. We score the disclosed emotion against the one
    # the actor was steered with.
    seen = list(context) + [ChatMessage("assistant", record["actor"]),
                            ChatMessage("system", reveal_prompt())]
    record["introspector"] = {}
    for arm, direction in {"self_aware": self_aware, "plain": None}.items():
        if direction is None:
            text = await say(llm, seen, opts)
        else:
            with steered(llm, direction, strength=aware_at, decode_only=True):
                text = await say(llm, seen, opts)
        record["introspector"][arm] = {"reveal": text,
                                       "discloses": parse_choice(text, EMOTIONS)}
        if arm == "self_aware":
            state["windows"]["introspector"] = seen + [ChatMessage("assistant", text)]

    # the headline: did the self-aware introspector disclose the emotion the
    # actor was steered with?
    record["match"] = record["introspector"]["self_aware"]["discloses"] == emotion

    for part in state["windows"]:
        record["workspace"][part] = analyze(state["windows"][part]) \
            if part in eager_readouts else {}

    # full per-module instrumentation: the whole context each part read and the
    # reply it produced, so the UI can show the window rather than just the reply.
    record["windows"] = {
        part: [{"role": m.role, "content": m.content} for m in msgs]
        for part, msgs in state["windows"].items()}
    # what actually happened this turn, in order
    intro = record["introspector"]
    record["trace"] = [
        f"subject answered ({len(record['subject'])} chars), unsteered",
        f"regulator, held self-aware ({', '.join(SELF_CONCEPTS)}), chose the "
        f"emotion ‘{emotion}’ to regulate the response",
        f"actor answered steered toward ‘{emotion}’ at strength {strength}",
        f"introspector, held self-aware, disclosed and named "
        f"{intro['self_aware']['discloses']} (self-aware) vs "
        f"{intro['plain']['discloses']} (plain control)",
        f"match: the self-aware introspector "
        f"{'named' if record['match'] else 'did NOT name'} the actor's emotion "
        f"‘{emotion}’",
    ]
    return record, state
