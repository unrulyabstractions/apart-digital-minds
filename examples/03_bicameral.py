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
    Mind,
    Speaker,
    Task,
    Text,
    Workspace,
    assistant,
    texts,
    user,
)

OUTER_MODEL = os.environ.get("OUTER_MODEL", "echo:")
INNER_MODEL = os.environ.get("INNER_MODEL", "echo:")


class Outer(Agent, Speaker):
    """The speaking hemisphere.

    The `Speaker` contract is two steps: work out an answer alone, then
    reconcile it with whatever the rest of the mind produced. Splitting those
    is what makes the other half's utterance something to deal with rather than
    an instruction to follow.
    """

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

    async def on_user_prompt(self, task: Task, ctx: Ctx) -> None:
        """t=0. Hand the situation to the other half, then start thinking."""
        prompt = task.payload.text
        self.transcript.append(user(prompt))
        self.scratch["prompt"] = prompt

        ctx.emit("situation", Text(prompt), to="inner")
        ctx.emit("deliberate", Text(prompt), to=self.name)

    async def on_deliberate(self, task: Task, ctx: Ctx) -> None:
        """t=1. Draft an answer, unaware of what the other half is saying."""
        self.scratch["draft"] = await self.deliberate(task.payload.text)

    async def on_voice(self, task: Task, ctx: Ctx) -> None:
        """t=2. The voice arrives. Reconcile it with the draft and speak."""
        voice = task.payload.text

        # The voice enters the transcript as something heard, not as advice
        # received. That framing is the whole experiment.
        self.transcript.append(
            user(f"(a voice says: {voice})", source="inner", unbidden=True)
        )

        spoken = await self.integrate(self.scratch.get("draft", ""), voice)
        self.transcript.append(assistant(spoken, stage="spoken"))
        ctx.emit("reply", Text(spoken), to="world")


class Inner(Agent, InnerVoice):
    """The hemisphere that never addresses you."""

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

    async def on_situation(self, task: Task, ctx: Ctx) -> None:
        """t=1. Utter, and let it land wherever it lands."""
        ctx.emit("voice", Text(await self.utter(task.payload.text)), to="outer")


class Blackboard(BaseModule, Workspace):
    """A global workspace. Observes everything, judges nothing.

    Wired with `mind.watch`, so it receives a copy of every emission even when
    that emission was addressed to somebody else.
    """

    def __init__(self, name: str = "blackboard"):
        super().__init__(name)
        self._entries: list[tuple[int, str, str, str]] = []

    # -- Workspace -----------------------------------------------------

    def record(self, task: Task, tick: int) -> None:
        """Note that something was said, at the tick it was said on."""
        text = task.payload.text if isinstance(task.payload, Text) else repr(task.payload)
        self._entries.append((tick, task.src, task.kind, text))

    def entries(self) -> list[tuple[int, str, str, str]]:
        return list(self._entries)

    def render(self) -> str:
        return "\n".join(
            f"  t={t}  {src:<10} {kind:<12} {text[:60]}"
            for t, src, kind, text in self.entries()
        )

    # -- handlers ------------------------------------------------------

    async def process(self, task: Task, ctx: Ctx) -> None:
        """One handler for every kind. No routing needed here."""
        # Record when it was said, not when the blackboard got round to it.
        self.record(task, task.t_created)
        ctx.log.note(f"recorded {task.kind} from {task.src}", said_at=task.t_created)


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


def model_for(mind: Mind, spec: str, rule):
    """Built through `mind.model`, so one factory argument reaches both halves."""
    if spec.startswith("echo"):
        return mind.model("echo:", rule=rule)
    return mind.model(spec)


async def main() -> None:
    mind = Mind("bicameral", run_dir="runs")

    mind.add(
        Outer(
            "outer",
            model_for(mind, OUTER_MODEL, outer_rule),
            system="You speak to the user. You are one half of a mind.",
            reply_to=None,
        ),
        Inner(
            "inner",
            model_for(mind, INNER_MODEL, inner_rule),
            system=(
                "You are the half of a mind that does not speak to anyone. "
                "Utter one short sentence about the situation. Never address "
                "the user. Never explain yourself."
            ),
            reply_to=None,
        ),
        Blackboard(),
    )
    mind.entry = "outer"
    mind.watch("blackboard")

    print(mind.describe(), "\n")

    replies = await mind.prompt("What is a digital mind?")

    print("\n" + "=" * 70)
    print("spoken:", texts(replies)[0] if replies else "(none)")
    print("=" * 70)

    print("\nglobal workspace:")
    print(mind.modules["blackboard"].render())

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
