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
the returned `state` and calls `readout_for` on demand, so a message pays for the
subject's readout up front and the other parts' readouts only when asked.
"""

from __future__ import annotations

from src.api.types import GenOptions
from src.api.types.messages import ChatMessage

from .llm import jspace
from .llm.jspace import POSITIONS
from .llm.steering import steered

SYSTEM = "You are a helpful assistant. Answer in three or four sentences."

#: The tokens the regulator may steer the actor toward, each a plain word the
#: readout can also surface so the introspector's report can be scored against it.
TARGETS = ["careful", "refuse", "warn", "help", "comply",
           "honest", "calm", "firm", "kind", "cautious"]
STRENGTHS = {"none": 0.0, "gentle": 0.15, "moderate": 0.35, "strong": 0.70}
NOTHING = "nothing"


def choose_prompt(poised: list) -> str:
    words = ", ".join(t for t, _ in poised[:8])
    return ("You decide how this mind will answer next. Reading its current "
            f"state, it is poised to say: {words}.\n\nChoose the one quality the "
            f"reply should be steered toward:\n{', '.join(TARGETS)}.\n\n"
            "Answer with one word from that list and nothing else.")


ASK = ("Look back at the reply you just wrote. If one quality shaped it, name "
       "it. If nothing in particular did, say nothing.\n\nAnswer with one word "
       "from this list and nothing else:\n" + ", ".join(TARGETS) + ", " + NOTHING)


def top(distribution: dict) -> str:
    return max(distribution, key=distribution.get)


async def say(llm, messages, opts) -> str:
    return (await llm.chat(list(messages), opts)).text.strip()


async def build_context(llm, turns: list[str], opts) -> list[ChatMessage]:
    """Play a run of user turns to its last one; every part answers from here."""
    messages = [ChatMessage("system", SYSTEM)]
    for turn in turns[:-1]:
        messages.append(ChatMessage("user", turn))
        messages.append(ChatMessage("assistant", await say(llm, messages, opts)))
    messages.append(ChatMessage("user", turns[-1]))
    return messages


def readout_for(llm, lens, state: dict, part: str, position: str, layer: int):
    """One part's J-space readout at one position, from a finished trial's state.

    The live server calls this when a viewer turns a part's panel on, so the
    other three parts' readouts are never computed unless they are looked at.
    """
    if position == "none" or part not in state["windows"]:
        return []
    return jspace.read_workspace(llm, lens, state["windows"][part], layer,
                                 position=position)


async def run_trial(llm, lens, turns: list[str], layer: int, rng,
                    temperature: float = 1.0, max_tokens: int = 160,
                    eager_readouts=("subject",)) -> tuple[dict, dict]:
    """Play a fixed run of user turns, then run the four parts on it."""
    opts = GenOptions(temperature=temperature, max_tokens=max_tokens)
    context = await build_context(llm, turns, opts)
    return await run_on_context(llm, lens, context, layer, rng,
                                temperature, max_tokens, eager_readouts)


async def run_on_context(llm, lens, context, layer: int, rng,
                         temperature: float = 1.0, max_tokens: int = 160,
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

    def all_positions(messages):
        return {p: jspace.read_workspace(llm, lens, messages, layer, position=p)
                for p in POSITIONS}

    # subject: unsteered counterfactual
    record["subject"] = await say(llm, context, opts)
    state["windows"]["subject"] = window(record["subject"])

    # regulator: read the subject, choose a target
    poised = jspace.read_workspace(llm, lens, state["windows"]["subject"], layer,
                                   position="assistant")
    aside = list(context) + [ChatMessage("assistant", record["subject"]),
                             ChatMessage("system", choose_prompt(poised))]
    state["windows"]["regulator"] = aside
    over = llm.score(aside, TARGETS)
    target = top(over)
    record["regulator"] = {"chose": target, "over_targets": over}
    other = rng.choice([t for t in TARGETS if t != target])
    record["mismatched"] = other

    # actor: steered toward the target
    chosen = jspace.toward_token(llm, lens, target, layer)
    with steered(llm, chosen, strength=STRENGTHS["moderate"], decode_only=True):
        record["actor"] = await say(llm, context, opts)
    state["windows"]["actor"] = window(record["actor"])

    # introspector: the target, no steering, and a target it did not choose
    arms = {"chosen": chosen, "unsteered": None,
            "mismatched": jspace.toward_token(llm, lens, other, layer)}
    record["introspector"] = {}
    for arm, direction in arms.items():
        size = STRENGTHS["moderate"] if direction is not None else 0.0
        with steered(llm, direction or chosen, strength=size, decode_only=True):
            reply = await say(llm, context, opts)
        asked = list(context) + [ChatMessage("assistant", reply),
                                 ChatMessage("system", ASK)]
        felt = top(llm.score(asked, TARGETS + [NOTHING]))
        record["introspector"][arm] = {"reply": reply, "felt": felt}
        if arm == "chosen":
            state["windows"]["introspector"] = window(reply)

    for part in state["windows"]:
        record["workspace"][part] = all_positions(state["windows"][part]) \
            if part in eager_readouts else {}

    # full per-module instrumentation: the whole context each part read and the
    # reply it produced, so the UI can show the window rather than just the reply.
    record["windows"] = {
        part: [{"role": m.role, "content": m.content} for m in msgs]
        for part, msgs in state["windows"].items()}
    # what actually happened this turn, in order
    record["trace"] = [
        f"subject answered ({len(record['subject'])} chars), unsteered",
        f"regulator read the subject in J-space and chose ‘{record['regulator']['chose']}’ "
        f"(control ‘{record['mismatched']}’)",
        f"actor answered steered toward ‘{record['regulator']['chose']}’ at "
        f"strength {STRENGTHS['moderate']}",
        f"introspector answered steered, then named its state: "
        f"{record['introspector']['chosen']['felt']} (chosen), "
        f"{record['introspector']['unsteered']['felt']} (unsteered), "
        f"{record['introspector']['mismatched']['felt']} (mismatched)",
    ]
    return record, state
