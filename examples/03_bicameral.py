"""A bicameral mind, assembled from the same parts as everything else.

Two hemispheres and a shared workspace, each declaring the role it plays:

    outer        Agent + Speaker. The half that speaks. It owns the
                 conversation, and must reconcile the other half's utterance
                 with what it was going to say anyway.
    inner        Agent + InnerVoice. The half that never addresses you. It
                 produces an unbidden voice about the situation, which arrives
                 at the outer half as something it then has to deal with.
    blackboard   Module + Workspace. It observes all traffic and keeps it.

The tick structure:

    t=0   outer takes the prompt, tells inner the situation, and schedules
          its own deliberation.
    t=1   both halves run at the same time. Neither can see the other's work,
          because emissions from tick 1 are not delivered until tick 2.
    t=2   outer hears the voice, reconciles it with its own draft, and speaks.

Tick 1 is where the async core earns its place: two models are in flight at
once, and the outcome still does not depend on which returns first.

Run it:
    python examples/03_bicameral.py
    OUTER_MODEL=anthropic:claude-opus-5 INNER_MODEL=ollama:qwen3:8b \
        python examples/03_bicameral.py
"""

from __future__ import annotations

import asyncio
import os

from src import (
    Agent,
    BaseModule,
    Ctx,
    InnerVoice,
    Message,
    Mind,
    Soul,
    Speaker,
    Text,
    Workspace,
    assistant,
    get_llm,
    texts,
    user,
)

OUTER_MODEL = os.environ.get("OUTER_MODEL", "echo:")
INNER_MODEL = os.environ.get("INNER_MODEL", "echo:")


class Outer(Soul, Speaker):
    """The speaking hemisphere.

    The `Speaker` contract is two steps: work out an answer alone, then
    reconcile it with whatever the rest of the mind produced. Splitting those
    is what makes the other half's utterance something to deal with rather than
    an instruction to follow.
    """

    # A custom soul: it publishes different things from the default one, so
    # it replaces OUTPUTS rather than extending them.
    OUTPUTS = {
        "situation": "what is going on, for the other half",
        "deliberate": "a note to itself, to draft on the next tick",
        "reply": "what it says out loud",
    }
    INPUTS = {"user_prompt": "a question", "voice": "the other half, uninvited"}

    # -- Speaker -------------------------------------------------------

    async def deliberate(self, prompt: str) -> str:
        """Work out an answer, without input from the rest of the mind."""
        completion = await self.think(
            messages=[*self.transcript.messages, user("[deliberate] Draft your answer.")],
            tag="draft",
        )
        return completion.text

    async def integrate(self, draft: str, voice: str) -> str:
        """Reconcile that draft with what the rest of the mind said."""
        completion = await self.think(
            messages=[
                *self.transcript.messages,
                user(f"[integrate] Your draft was: {draft}\nNow answer."),
            ],
            tag="speak",
        )
        return completion.text

    # -- handlers ------------------------------------------------------

    async def on_user_prompt(self, message: Message, ctx: Ctx) -> None:
        """t=0. Hand the situation to the other half, then start thinking."""
        prompt = message.payload.text
        self.transcript.append(user(prompt))
        self.scratch["prompt"] = prompt

        ctx.emit("situation", Text(prompt))
        ctx.emit("deliberate", Text(prompt))  # registered back to itself

    async def on_deliberate(self, message: Message, ctx: Ctx) -> None:
        """t=1. Draft an answer, unaware of what the other half is saying."""
        self.scratch["draft"] = await self.deliberate(message.payload.text)

    async def on_voice(self, message: Message, ctx: Ctx) -> None:
        """t=2. The voice arrives. Reconcile it with the draft and speak."""
        voice = message.payload.text

        # The voice enters the transcript as something heard, not as advice
        # received. That framing is the whole experiment.
        self.transcript.append(
            user(f"(a voice says: {voice})", source="inner", unbidden=True)
        )

        spoken = await self.integrate(self.scratch.get("draft", ""), voice)
        self.transcript.append(assistant(spoken, stage="spoken"))
        ctx.emit("reply", Text(spoken))


class Inner(Agent, InnerVoice):
    """The hemisphere that never addresses you."""

    OUTPUTS = {"voice": "an unbidden utterance about the situation"}
    INPUTS = {"situation": "what is going on"}

    # -- InnerVoice ----------------------------------------------------

    async def utter(self, situation: str) -> str:
        """Say something about the situation, to no one in particular."""
        completion = await self.think(
            messages=[
                *self.transcript.messages,
                user(f"[voice] The situation is: {situation}"),
            ],
            tag="voice",
        )
        return completion.text.strip()

    # -- handlers ------------------------------------------------------

    async def on_situation(self, message: Message, ctx: Ctx) -> None:
        """t=1. Utter, and let it land wherever it lands."""
        ctx.emit("voice", Text(await self.utter(message.payload.text)))


class Blackboard(BaseModule, Workspace):
    """A global workspace. Observes everything, judges nothing.

    Registered onto `"*"` of both halves, so it receives a copy of everything
    either of them emits, whoever it was meant for.
    """

    INPUTS = {"*": "anything either hemisphere emits"}

    def __init__(self, name: str = "blackboard"):
        super().__init__(name)
        self._entries: list[tuple[int, str, str, str]] = []

    # -- Workspace -----------------------------------------------------

    def record(self, message: Message, tick: int) -> None:
        """Note that something was said, at the tick it was said on."""
        payload = message.payload
        text = payload.text if isinstance(payload, Text) else repr(payload)
        self._entries.append((tick, message.src, message.channel, text))

    def entries(self) -> list[tuple[int, str, str, str]]:
        return list(self._entries)

    def render(self) -> str:
        return "\n".join(
            f"  t={t}  {src:<10} {kind:<12} {text[:60]}"
            for t, src, kind, text in self.entries()
        )

    # -- handlers ------------------------------------------------------

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        """One handler for every channel. It never emits, so it never wires."""
        # Record when it was said, not when the blackboard got round to it.
        self.record(message, message.t_created)
        ctx.log.note(
            f"recorded {message.channel} from {message.src}",
            said_at=message.t_created,
        )


# -- stand-in models -------------------------------------------------------


def outer_rule(messages, opts) -> str:
    last = messages[-1].content if messages else ""
    if "[deliberate]" in last:
        return "A digital mind is a system that models itself well enough to be surprised."
    if "[integrate]" in last:
        heard = next(
            (m.content for m in reversed(messages) if m.meta.get("unbidden")), ""
        )
        return (
            "A digital mind is a system that models itself well enough to be "
            f"surprised. And something in me insists: {heard[len('(a voice says: '):-1]}"
        )
    return "..."


def inner_rule(messages, opts) -> str:
    return "you are describing yourself"


def stand_in(spec: str, rule):
    """The real model if one was named, otherwise the scripted fake."""
    return get_llm("echo:", rule=rule) if spec.startswith("echo") else get_llm(spec)


async def main() -> None:
    # The speaking half is the soul: it is the model this experiment is about.
    # `soul=Outer` puts a custom one at the centre instead of the default.
    mind = Mind(
        "bicameral",
        stand_in(OUTER_MODEL, outer_rule),
        system="You speak to the user. You are one half of a mind.",
        soul=Outer,
        run_dir="runs",
    )
    outer = mind.soul
    inner = Inner(
        "inner",
        stand_in(INNER_MODEL, inner_rule),
        system=(
            "You are the half of a mind that does not speak to anyone. "
            "Utter one short sentence about the situation. Never address "
            "the user. Never explain yourself."
        ),
    )
    blackboard = Blackboard()

    # The soul already speaks to you. The rest of the mind is wired by hand.
    outer.register(inner, "situation")
    outer.register(outer, "deliberate")  # a note to itself, heard next tick
    inner.register(outer, "voice")

    # The workspace listens to everything either half emits.
    outer.register(blackboard, "*")
    inner.register(blackboard, "*")

    print(mind.describe(), "\n")

    mind.prompt("What is a digital mind?")
    await mind.process()
    replies = mind.get_replies()

    print("\n" + "=" * 70)
    print("spoken:", texts(replies)[0] if replies else "(none)")
    print("=" * 70)

    print("\nglobal workspace:")
    print(blackboard.render())

    print(f"\nticks used: {mind.scheduler.t}")

    # Both halves ran inside tick 1. Prove it from the trace.
    calls = [
        (e.tick, e.module)
        for e in mind.events.events
        if e.kind == "llm.request"
    ]
    print(f"model calls by tick: {calls}")

    mind.close()


if __name__ == "__main__":
    asyncio.run(main())
