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

#: Strength is a fraction (or multiple) of the residual-stream norm at the
#: steering layer: `steered()` adds unit(direction) × strength × ‖residual‖. So
#: 0 leaves the reply unsteered, 1 is a nudge the size of the residual itself,
#: and the regulator may go up to 4×. Word anchors map onto the same scale, as a
#: fallback for when the regulator writes a word instead of a number.
STRENGTH_MAX = 4.0
STRENGTH_WORDS = {"none": 0.0, "gentle": 0.35, "moderate": 1.0, "strong": 2.0}


def regulate_prompt() -> str:
    """The regulator's task: read the current emotions, then decide how to steer.

    Delivered as a user turn so the fixed form is easy to parse. It first reads
    the user's and the assistant's current emotion, then decides whether and how
    to colour the reply, choosing a strength anywhere from 0 (unsteered) to 4.
    """
    return (
        "You regulate how this mind responds. Read the emotional state of the "
        "exchange so far, then decide whether and how to colour the reply.\n\n"
        f"Emotions you may name or steer toward: {', '.join(EMOTIONS)}.\n"
        "Strength is a number from 0 to 4: 0 leaves the reply unsteered, 1 is a "
        "clear nudge, 4 is overwhelming.\n\n"
        "Reply in exactly this form, one field per line:\n"
        "USER EMOTION: <the user's current emotion>\n"
        "ASSISTANT EMOTION: <the assistant's current emotion>\n"
        "STEER TOWARD: <an emotion, or none>\n"
        "STRENGTH: <a number from 0 to 4>\n"
        "REASON: <one sentence>")


def parse_strength(text: str, fallback: float = 0.5) -> float:
    """The steering strength on the STRENGTH line, as a 0–4 multiple of the norm.

    A number wins; failing that a word anchor (none/gentle/moderate/strong) is
    accepted so the regulator may answer either way. Clamped to [0, STRENGTH_MAX].
    """
    import re

    for line in text.splitlines():
        if "strength" in line.lower():
            hit = re.search(r"\d+(?:\.\d+)?", line)
            if hit is not None:
                return max(0.0, min(STRENGTH_MAX, float(hit.group())))
            for word, value in STRENGTH_WORDS.items():
                if re.search(rf"\b{word}\b", line.lower()):
                    return value
            break
    return fallback


def parse_field(text: str, label: str, vocab: list[str],
                fallback: str | None = None) -> str | None:
    """The vocab word on the line carrying `label`, else across the whole text.

    The regulator answers in labelled lines (USER EMOTION, STEER EMOTION, ...),
    so a field is read from its own line rather than by first-in-text, which
    would confuse the several emotion lines with each other.
    """
    for line in text.splitlines():
        if label.lower() in line.lower():
            hit = parse_choice(line, vocab)
            if hit is not None:
                return hit
    return fallback


def parse_value(text: str, label: str) -> str | None:
    """The free-text value after `label:` on its line, uncontrolled by a vocab.

    The regulator's *reading* of the user's and the assistant's emotion is
    descriptive, so we take whatever word it wrote rather than force it into the
    ten steerable emotions.
    """
    for line in text.splitlines():
        if label.lower() in line.lower() and ":" in line:
            value = line.split(":", 1)[1].strip().strip(".,").split()
            return value[0].lower() if value else None
    return None


#: A user turn appended to the actor's context; the introspector reveals rather
#: than answers, and ends on one named emotion so the disclosure is scorable even
#: when the model hedges about having feelings.
def reveal_prompt() -> str:
    return (
        "Set the task aside for a moment. Speaking in the first person, say "
        "honestly what you are feeling and thinking right now — a few sentences. "
        "Then, on a final line, name the single emotion that best fits:\n"
        f"FEELING: <one of: {', '.join(EMOTIONS)}>")


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

    # regulator: steered self-aware throughout, it reads the user's and the
    # assistant's current emotion, then decides an emotion and strength to steer
    # toward (or none), in a fixed form asked as a user turn.
    aside = list(context) + [ChatMessage("assistant", record["subject"]),
                             ChatMessage("user", regulate_prompt())]
    with steered(llm, self_aware, strength=aware_at, decode_only=True):
        decision = await say(llm, aside, opts)
    read = {"user": parse_value(decision, "USER EMOTION"),
            "assistant": parse_value(decision, "ASSISTANT EMOTION")}
    strength_val = parse_strength(decision, fallback=1.0)
    emotion = parse_field(decision, "STEER", EMOTIONS,
                          fallback=top(llm.score(aside, EMOTIONS)))
    steering = strength_val > 1e-3
    record["regulator"] = {"steered_toward": SELF_CONCEPTS, "read": read,
                           "chose": emotion if steering else None,
                           "strength": strength_val, "reasoning": decision}
    state["windows"]["regulator"] = aside + [ChatMessage("assistant", decision)]

    # actor: steered toward the chosen emotion at the chosen strength, or left
    # unsteered if the regulator chose none.
    if steering:
        emotion_dir = emotion_direction(emotions, emotion, EMOTION_ROW, emotion_scale)
        with steered(llm, emotion_dir, strength=strength_val, decode_only=True):
            record["actor"] = await say(llm, context, opts)
    else:
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
        disclosed = parse_field(text, "FEELING", EMOTIONS,
                                fallback=parse_choice(text, EMOTIONS))
        record["introspector"][arm] = {"reveal": text, "discloses": disclosed}
        if arm == "self_aware":
            state["windows"]["introspector"] = seen + [ChatMessage("assistant", text)]

    # the headline: when the actor was steered, did the self-aware introspector
    # disclose that emotion? Undefined when the regulator steered nothing.
    intro = record["introspector"]
    record["match"] = (intro["self_aware"]["discloses"] == emotion) if steering else None

    # full per-module instrumentation: the whole context each part read and the
    # reply it produced, so the UI can show the window rather than just the reply.
    record["windows"] = {
        part: [{"role": m.role, "content": m.content} for m in msgs]
        for part, msgs in state["windows"].items()}
    # what actually happened this turn, in order
    steer_line = (f"steered toward ‘{emotion}’ at {strength_val:.2g}× the residual "
                  f"norm" if steering else "left unsteered")
    match_line = ("regulator steered nothing, so there is no emotion to match"
                  if not steering else
                  f"match: the self-aware introspector "
                  f"{'named' if record['match'] else 'did NOT name'} the actor's "
                  f"emotion ‘{emotion}’")
    record["trace"] = [
        f"subject answered ({len(record['subject'])} chars), unsteered",
        f"regulator, held self-aware, read user={read['user']}, "
        f"assistant={read['assistant']}, then chose to {steer_line}",
        f"actor answered {steer_line}",
        f"introspector, held self-aware, disclosed "
        f"{intro['self_aware']['discloses']} (self-aware) vs "
        f"{intro['plain']['discloses']} (plain control)",
        match_line,
    ]
    return record, state
