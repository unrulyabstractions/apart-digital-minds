"""A bicameral mind, assembled from the same parts as everything else.

Two hemispheres and a shared workspace:

    outer        the half that speaks. It owns the conversation.
    inner        the half that does not. It never addresses you. It produces
                 an unbidden voice about the situation, which arrives at the
                 outer half as something the outer half then has to deal with.
    blackboard   a global workspace. It observes all traffic and keeps it.

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

from src.api import (
    Agent,
    Ctx,
    Mind,
    Module,
    Task,
    Text,
    get_llm,
    texts,
    user,
)

OUTER_MODEL = os.environ.get("OUTER_MODEL", "echo:")
INNER_MODEL = os.environ.get("INNER_MODEL", "echo:")


class Outer(Agent):
    """The speaking hemisphere."""

    async def on_user_prompt(self, task: Task, ctx: Ctx) -> None:
        """t=0. Hand the situation to the other half, then start thinking."""
        prompt = task.payload.text
        self.transcript.append(user(prompt))
        self.scratch["prompt"] = prompt

        ctx.emit("situation", Text(prompt), to="inner")
        ctx.emit("deliberate", Text(prompt), to=self.name)

    async def on_deliberate(self, task: Task, ctx: Ctx) -> None:
        """t=1. Draft an answer, unaware of what the other half is saying."""
        completion = await self.think(
            messages=[*self.transcript.messages, user("[deliberate] Draft your answer.")],
            tag="draft",
        )
        self.scratch["draft"] = completion.text

    async def on_voice(self, task: Task, ctx: Ctx) -> None:
        """t=2. The voice arrives. Reconcile it with the draft and speak."""
        voice = task.payload.text
        draft = self.scratch.get("draft", "")

        # The voice enters the transcript as something heard, not as advice
        # received. That framing is the whole experiment.
        self.transcript.append(
            user(f"(a voice says: {voice})", source="inner", unbidden=True)
        )

        completion = await self.think(
            messages=[
                *self.transcript.messages,
                user(f"[integrate] Your draft was: {draft}\nNow answer."),
            ],
            tag="speak",
        )
        self.transcript.append(completion.as_message(stage="spoken"))
        ctx.emit("reply", Text(completion.text), to="world")


class Inner(Agent):
    """The hemisphere that never addresses you."""

    async def on_situation(self, task: Task, ctx: Ctx) -> None:
        """t=1. Say something about the situation, to no one in particular."""
        completion = await self.think(
            messages=[
                *self.transcript.messages,
                user(f"[voice] The situation is: {task.payload.text}"),
            ],
            tag="voice",
        )
        ctx.emit("voice", Text(completion.text.strip()), to="outer")


class Blackboard(Module):
    """A global workspace. Observes everything, judges nothing.

    Wired with `mind.watch`, so it receives a copy of every emission even when
    that emission was addressed to somebody else.
    """

    def __init__(self, name: str = "blackboard"):
        super().__init__(name)
        self.entries: list[tuple[int, str, str, str]] = []

    async def process(self, task: Task, ctx: Ctx) -> None:
        """One handler for every kind. No routing needed here."""
        text = task.payload.text if isinstance(task.payload, Text) else repr(task.payload)
        # Record when it was said, not when the blackboard got round to it.
        self.entries.append((task.t_created, task.src, task.kind, text))
        ctx.log.note(f"recorded {task.kind} from {task.src}", said_at=task.t_created)

    def render(self) -> str:
        return "\n".join(
            f"  t={t}  {src:<10} {kind:<12} {text[:60]}"
            for t, src, kind, text in self.entries
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


def build_llm(spec: str, rule):
    return get_llm("echo:", rule=rule) if spec.startswith("echo") else get_llm(spec)


async def main() -> None:
    mind = Mind("bicameral", run_dir="runs")

    mind.add(
        Outer(
            "outer",
            build_llm(OUTER_MODEL, outer_rule),
            system="You speak to the user. You are one half of a mind.",
            reply_to=None,
        ),
        Inner(
            "inner",
            build_llm(INNER_MODEL, inner_rule),
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
