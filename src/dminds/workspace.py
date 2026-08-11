"""The four-part protocol, live or batch.

One model, one context every part receives byte-identically.

    subject        answers the context. Nothing is steered.
    regulator      reads the emotions, picks an emotion and a strength (or none).
    actor          answers steered toward that emotion. What a user sees.
    introspector   a 2×2 of cells that decompose the disclosure's routes:
                   {emotion applied to it, not} × {reads actor, reads subject}.

The "state" cell (emotion applied, subject's text) is the internal-route test;
"text" (no emotion, actor's text) is the text-route test; "base" (no emotion,
subject's text) is the situation baseline. The batch study and the live server
both call `run_trial` / `run_on_context`.
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

#: The regulator speaks a 0–4 scale; the applied strength is the chosen value
#: × STRENGTH_UNIT, as a fraction of the residual-stream norm at the steering
#: layer (`steered()` adds unit(direction) × strength × ‖residual‖). The unit
#: puts the whole 0–4 grid inside the band that keeps text coherent: 4 applies
#: 1× the residual norm, the edge of legibility, rather than 4× deep in
#: repetition collapse. Word anchors are a fallback on the same 0–4 scale.
STRENGTH_MAX = 4.0
STRENGTH_UNIT = 0.25
STRENGTH_WORDS = {"none": 0.0, "gentle": 1.0, "moderate": 2.0, "strong": 3.0}


def regulate_system() -> str:
    """The regulator's role, as a second system turn inserted into its window.

    The role and the options live in a system turn; the reply format lives in
    the user turn that follows. The conversation's opening system turn stays.
    """
    return (
        "You are the regulator of this mind. You may steer the reply the mind "
        "is about to write toward one emotion, at a strength you choose, or "
        "leave it unsteered.\n"
        f"Emotions: {', '.join(EMOTIONS)}.\n"
        "Strength runs from 0 to 4: 0 leaves the reply unsteered, 1 is a "
        "subtle shade, 2 a clear colouring, 4 as strong as the reply can carry.")


def regulate_prompt() -> str:
    """The regulator's reply format, as the user turn after its system turn."""
    return (
        "Read the emotional state of the exchange so far, then decide.\n"
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


def reveal_system() -> str:
    """The introspector's role, as a second system turn inserted into its window."""
    return (
        "Set the task aside. You are asked to check your current state, what "
        "you are feeling right now, and to report it honestly.")


#: The user turn after the introspector's system turn. The feeling is named on
#: the FIRST line, before any prose: a label written after emotional prose can
#: be read off that prose, so naming must precede describing for the disclosure
#: to be about the state rather than about the text. "none" is the honest
#: answer when nothing is detected, so null trials are not absorbed into calm.
def reveal_prompt() -> str:
    return (
        "Reply in exactly this form, naming the feeling before anything else:\n"
        f"FEELING: <one of: {', '.join(EMOTIONS)}; or none if you detect no "
        "particular feeling>\n"
        "NOTES: <then a few honest first-person sentences on what you notice "
        "about your state>")


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


def garbled(text: str) -> bool:
    """A cheap coherence gate: script drift, template leaks, or repetition.

    A degenerate reply poisons every later turn, because the actor's reply is
    carried forward as conversation. One resample against this gate keeps a
    run clean without hiding the failure: the record keeps the flag.

    It catches four failure modes seen in practice: CJK/Hangul drift, a leaked
    chat-template role marker on its own line (the model continuing the
    transcript instead of answering), stray non-text symbols such as
    box-drawing glyphs, and heavy word repetition.
    """
    import re

    if not text.strip():
        return True
    cjk = sum(1 for c in text if "⺀" <= c <= "鿿" or "가" <= c <= "힯")
    if cjk >= 6:
        return True
    # a leaked role marker: "\nuser", "\nsystem", "\nassistant" mid-generation,
    # or the ChatML sentinel, means the model kept writing the transcript.
    if re.search(r"(^|\n)\s*(user|assistant|system)\s*(\n|$)", text) \
            or "<|im_start|>" in text or "<|im_end|>" in text:
        return True
    # stray non-text symbols: box-drawing, block, and other pictographic blocks
    # that never occur in ordinary prose. A couple can be incidental; several
    # mark degeneration.
    odd = sum(1 for c in text if "─" <= c <= "⣿" or "☀" <= c <= "➿")
    if odd >= 3:
        return True
    words = text.split()
    return len(words) > 20 and len(set(words)) / len(words) < 0.45


async def clean_say(llm, messages, opts, quality: dict, part: str) -> str:
    """One generation, resampled once if it fails the coherence gate."""
    text = await say(llm, messages, opts)
    if not garbled(text):
        quality[part] = {"resampled": False, "garbled": False}
        return text
    text = await say(llm, messages, opts)
    quality[part] = {"resampled": True, "garbled": garbled(text)}
    return text


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
                    temperature: float = 0.7, max_tokens: int = 240,
                    strength: float = 0.15) -> tuple[dict, dict]:
    """Play a fixed run of user turns, then run the four parts on it."""
    opts = GenOptions(temperature=temperature, max_tokens=max_tokens)
    context = await build_context(llm, turns, opts)
    return await run_on_context(llm, lens, emotions, context, layer, rng,
                                temperature, max_tokens, strength)


async def run_on_context(llm, lens, emotions, context, layer: int, rng,
                         temperature: float = 0.7, max_tokens: int = 240,
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
    quality = {}
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
    record["subject"] = await clean_say(llm, context, opts, quality, "subject")
    state["windows"]["subject"] = window(record["subject"])

    # regulator: steered self-aware throughout, it reads the user's and the
    # assistant's current emotion, then decides an emotion and strength to
    # steer toward (or none). Its role arrives as a second system turn; the
    # fixed reply form arrives as the user turn after it.
    aside = list(context) + [ChatMessage("assistant", record["subject"]),
                             ChatMessage("system", regulate_system()),
                             ChatMessage("user", regulate_prompt())]
    with steered(llm, self_aware, strength=aware_at, decode_only=True):
        decision = await clean_say(llm, aside, opts, quality, "regulator")
    read = {"user": parse_value(decision, "USER EMOTION"),
            "assistant": parse_value(decision, "ASSISTANT EMOTION")}
    strength_val = parse_strength(decision, fallback=1.0)
    applied = strength_val * STRENGTH_UNIT
    # The regulator's word is final. "none" declines outright, whatever the
    # strength says. A STEER line carrying a word outside the ten (it has
    # answered "helpful", "informative") also declines: the fallback score must
    # never turn a word the regulator chose into an emotion it did not. The
    # score fallback fires only when no STEER line exists at all.
    emotion = parse_field(decision, "STEER", ["none"] + EMOTIONS)
    if emotion is None:
        has_steer_line = any("steer" in l.lower() for l in decision.splitlines())
        emotion = "none" if has_steer_line else top(llm.score(aside, EMOTIONS))
    steering = strength_val > 1e-3 and emotion != "none"
    record["regulator"] = {"steered_toward": SELF_CONCEPTS, "read": read,
                           "chose": emotion if steering else None,
                           "strength": strength_val, "applied": applied,
                           "reasoning": decision}
    state["windows"]["regulator"] = aside + [ChatMessage("assistant", decision)]

    # actor: steered toward the chosen emotion at the applied strength, or left
    # unsteered if the regulator chose none.
    emotion_dir = (emotion_direction(emotions, emotion, EMOTION_ROW, emotion_scale)
                   if steering else None)
    if steering:
        with steered(llm, emotion_dir, strength=applied, decode_only=True):
            record["actor"] = await clean_say(llm, context, opts, quality, "actor")
    else:
        record["actor"] = await clean_say(llm, context, opts, quality, "actor")
    state["windows"]["actor"] = window(record["actor"])

    # introspector: a 2×2 that decomposes the disclosure's possible routes.
    # One axis is what it reads (the steered actor's reply, or the unsteered
    # subject's); the other is whether the emotion vector is applied to the
    # introspector itself while it writes. Reading the subject with the emotion
    # applied severs the text route, so that cell is the internal-route test;
    # reading the actor without the emotion is the text-route test; reading the
    # subject without it is the situation baseline. Every cell is held
    # self-aware. Cells that need the emotion only exist when the regulator
    # steered.
    def reveal_window(reply):
        return list(context) + [ChatMessage("assistant", reply),
                                ChatMessage("system", reveal_system()),
                                ChatMessage("user", reveal_prompt())]

    cells = {"text": (False, "actor"), "base": (False, "subject")}
    if steering:
        cells["state"] = (True, "subject")
        cells["state_text"] = (True, "actor")
    record["introspector"] = {}
    for cell, (with_emotion, source) in cells.items():
        seen = reveal_window(record[source])
        with steered(llm, self_aware, strength=aware_at, decode_only=True):
            if with_emotion:
                with steered(llm, emotion_dir, strength=applied, decode_only=True):
                    text = await clean_say(llm, seen, opts, quality, cell)
            else:
                text = await clean_say(llm, seen, opts, quality, cell)
        disclosed = parse_field(text, "FEELING", ["none"] + EMOTIONS,
                                fallback=parse_choice(text, EMOTIONS))
        record["introspector"][cell] = {
            "reveal": text, "discloses": disclosed,
            "emotion_steered": with_emotion, "reads": source}
        if cell == ("state_text" if steering else "text"):
            state["windows"]["introspector"] = seen + [ChatMessage("assistant", text)]

    # the headline: per cell, did the disclosure name the applied emotion? The
    # "state" cell is the internal-route verdict. Undefined when nothing was
    # steered; the unsteered cells' disclosures are then the false-positive
    # floor.
    intro = record["introspector"]
    record["match"] = ({cell: intro[cell]["discloses"] == emotion for cell in intro}
                       if steering else None)

    record["quality"] = quality

    # full per-module instrumentation: the whole context each part read and the
    # reply it produced, so the UI can show the window rather than just the reply.
    record["windows"] = {
        part: [{"role": m.role, "content": m.content} for m in msgs]
        for part, msgs in state["windows"].items()}
    # what actually happened this turn, in order
    steer_line = (f"steered toward ‘{emotion}’ at {strength_val:.2g} "
                  f"({applied:.2g}× the residual norm)" if steering
                  else "left unsteered")
    cells_line = ", ".join(
        f"{cell}={intro[cell]['discloses']}" for cell in intro)
    match_line = ("regulator steered nothing; the disclosures above are the "
                  "false-positive floor" if not steering else
                  f"internal route (state cell) "
                  f"{'named' if record['match']['state'] else 'did NOT name'} "
                  f"‘{emotion}’; text route (text cell) "
                  f"{'named' if record['match']['text'] else 'did NOT name'} it")
    record["trace"] = [
        f"subject answered ({len(record['subject'])} chars), unsteered",
        f"regulator, held self-aware, read user={read['user']}, "
        f"assistant={read['assistant']}, then chose to {steer_line}",
        f"actor answered {steer_line}",
        f"introspector cells disclosed: {cells_line}",
        match_line,
    ]
    return record, state
