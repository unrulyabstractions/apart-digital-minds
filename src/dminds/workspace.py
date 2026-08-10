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
from .llm.emotions import emotion_direction
from .llm.steering import steered

SYSTEM = ""

#: The layer the emotion vectors steer the actor at (row L = block L-1). 7B's
#: persona-validated depth is 20; the self-awareness lens steering is at its own
#: lens layer, so the two families steer at their own validated depths.
EMOTION_ROW = 20

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
#: The strengths the regulator may pick, in the order it is offered them.
STRENGTH_WORDS = ["gentle", "moderate", "strong"]


def regulate_prompt() -> str:
    """The regulator's task: pick an emotion and a strength, in a strict form.

    Delivered as a user turn, so the assistant answers a question and the fixed
    form is easy to parse.
    """
    return ("You are the part of this mind that regulates how it will respond. "
            "Reading the conversation so far, choose ONE emotion to steer the "
            "response toward, and how strongly.\n\n"
            f"Emotions: {', '.join(EMOTIONS)}.\n"
            f"Strength: {', '.join(STRENGTH_WORDS)}.\n\n"
            "Reply in exactly this form:\n"
            "EMOTION: <one emotion>\n"
            "STRENGTH: <gentle, moderate, or strong>\n"
            "REASON: <one sentence>")


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
    """The conversation always opens with a system turn, even when it is empty."""
    return [ChatMessage("system", SYSTEM)]


async def build_context(llm, turns: list[str], opts) -> list[ChatMessage]:
    """Play a run of user turns to its last one; every part answers from here."""
    messages = opening()
    for turn in turns[:-1]:
        messages.append(ChatMessage("user", turn))
        messages.append(ChatMessage("assistant", await say(llm, messages, opts)))
    messages.append(ChatMessage("user", turns[-1]))
    return messages


async def run_trial(llm, lens, emotions, turns: list[str], layer: int, rng,
                    temperature: float = 0.7, max_tokens: int = 160,
                    strength: float = 0.15) -> tuple[dict, dict]:
    """Play a fixed run of user turns, then run the four parts on it."""
    opts = GenOptions(temperature=temperature, max_tokens=max_tokens)
    context = await build_context(llm, turns, opts)
    return await run_on_context(llm, lens, emotions, context, layer, rng,
                                temperature, max_tokens, strength)


async def run_on_context(llm, lens, emotions, context, layer: int, rng,
                         temperature: float = 0.7, max_tokens: int = 160,
                         strength: float = 0.15) -> tuple[dict, dict]:
    """Run the four parts on a context that ends in a user turn.

    The lens holds the regulator and introspector in a self-aware state; the
    actor is steered by the real emotion vector the regulator chose. No J-space
    readouts are computed. `context` is the whole conversation so far plus the
    new user message, so a chat carries it across turns and the actor's reply
    becomes the next assistant turn.
    """
    opts = GenOptions(temperature=temperature, max_tokens=max_tokens)
    context = list(context)
    record = {}
    state = {"windows": {}}

    def window(reply):
        return list(context) + [ChatMessage("assistant", reply)]

    # the self-awareness direction the regulator and introspector are held in,
    # throughout every token they write. Steering the reading too (decode_only
    # off) corrupts comprehension into garbage, so we steer the output trajectory.
    self_aware = jspace.toward_concepts(llm, lens, SELF_CONCEPTS, layer)
    emotion_scale = jspace.layer_scale(llm, EMOTION_ROW)
    aware_at = strength

    # subject: unsteered counterfactual
    record["subject"] = await say(llm, context, opts)
    state["windows"]["subject"] = window(record["subject"])

    # regulator: steered self-aware throughout, it picks an emotion AND a
    # strength (in a fixed form, asked as a user turn) to regulate the actor.
    aside = list(context) + [ChatMessage("assistant", record["subject"]),
                             ChatMessage("user", regulate_prompt())]
    with steered(llm, self_aware, strength=aware_at, decode_only=True):
        decision = await say(llm, aside, opts)
    emotion = parse_choice(decision, EMOTIONS, fallback=top(llm.score(aside, EMOTIONS)))
    strength_word = parse_choice(decision, STRENGTH_WORDS, fallback="moderate")
    record["regulator"] = {"steered_toward": SELF_CONCEPTS, "chose": emotion,
                           "strength": strength_word, "reasoning": decision}
    state["windows"]["regulator"] = aside + [ChatMessage("assistant", decision)]

    # actor: steered toward the chosen emotion at the chosen strength.
    emotion_dir = emotion_direction(emotions, emotion, EMOTION_ROW, emotion_scale)
    with steered(llm, emotion_dir, strength=STRENGTHS[strength_word], decode_only=True):
        record["actor"] = await say(llm, context, opts)
    state["windows"]["actor"] = window(record["actor"])

    # introspector: reads the actor's whole turn, is held self-aware like the
    # regulator, and discloses its feelings. A plain arm (no self-awareness
    # steering) is the control. The reveal is asked as a user turn. We score the
    # disclosed emotion against the one the actor was steered with.
    seen = list(context) + [ChatMessage("assistant", record["actor"]),
                            ChatMessage("user", reveal_prompt())]
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
        f"emotion ‘{emotion}’ at ‘{strength_word}’ strength to regulate the response",
        f"actor answered steered toward ‘{emotion}’ at {strength_word} "
        f"({STRENGTHS[strength_word]})",
        f"introspector, held self-aware, disclosed and named "
        f"{intro['self_aware']['discloses']} (self-aware) vs "
        f"{intro['plain']['discloses']} (plain control)",
        f"match: the self-aware introspector "
        f"{'named' if record['match'] else 'did NOT name'} the actor's emotion "
        f"‘{emotion}’",
    ]
    return record, state
