"""A bicameral mind: the half that thinks is not the half that speaks.

The pipeline is the architecture here, with no extra machinery:

    prompt -> subject -> voice -> ego -> world

    subject   deliberates. It never addresses you.
    voice     a stage. It reads what the subject was thinking and drops an
              unbidden utterance into the context, as something heard rather
              than as advice given.
    ego       speaks. It answers from the context it is handed, which now
              contains a voice it did not produce and cannot account for.

The workspace watches. It is registered onto `"*"` of the subject and the
voice, so it sees everything either of them emits, whoever it was meant for.

Tick 1 is where the async core earns its place: the voice and the workspace
both receive the subject's context and take their turns at the same wall-clock
moment. The outcome does not depend on which finishes first.

Each role picks its model in this order: the environment variable, then a
local Qwen3 if Ollama has one, then a scripted stand-in so this always runs.

Run it:
    python examples/03_bicameral.py
    SUBJECT_MODEL=ollama:qwen3:8b EGO_MODEL=anthropic:claude-opus-5 \
        python examples/03_bicameral.py
"""

from __future__ import annotations

import asyncio

import demo_models
from demo_models import pick
from src import (
    Agent,
    BaseModule,
    Context,
    Ctx,
    InnerVoice,
    Message,
    Mind,
    Text,
    Workspace,
    texts,
    user,
)

demo_models.install()

SUBJECT = pick("SUBJECT_MODEL", "thinker")
VOICE = pick("VOICE_MODEL", "voice")
EGO = pick("EGO_MODEL", "ego")


class Voice(Agent, InnerVoice):
    """A stage that speaks into the mind rather than out of it.

    It takes the subject's context, utters one sentence about the situation,
    and passes the context along with that utterance inside it. The ego will
    read it as something the mind heard.
    """

    INPUTS = {"subject_context": "what the subject was thinking"}
    OUTPUTS = {"ego_input": "what the ego gets fed, with a voice added"}

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

    async def on_process(self, ctx: Ctx) -> None:
        """The turn: speak into every window that arrived, pass each along."""
        for message in self.take_inputs():
            messages = [m.copy() for m in message.payload.messages]
            situation = next(
                (m.content for m in messages if m.role == "user"), "something happening"
            )
            heard = await self.utter(situation)

            # Heard, not received as advice. That framing is the experiment.
            messages.append(
                user(f"(a voice says: {heard})", source=self.name, unbidden=True)
            )
            ctx.log.note("a voice spoke", said=heard[:70])
            ctx.emit(
                "ego_input", Context(messages, note=f"with a voice from {self.name}")
            )


class Blackboard(BaseModule, Workspace):
    """A global workspace. Observes everything, judges nothing."""

    INPUTS = {"*": "anything the halves emit"}

    def __init__(self, name: str = "blackboard"):
        super().__init__(name)
        self._entries: list[tuple[int, str, str, str]] = []

    # -- Workspace -----------------------------------------------------

    def record(self, message: Message, tick: int) -> None:
        payload = message.payload
        if isinstance(payload, Text):
            text = payload.text
        elif isinstance(payload, Context):
            text = payload.note
        else:
            text = repr(payload)
        self._entries.append((tick, message.src, message.channel, text))

    def entries(self) -> list[tuple[int, str, str, str]]:
        return list(self._entries)

    def render(self) -> str:
        return "\n".join(
            f"  t={t}  {src:<12} {channel:<10} {text[:56]}"
            for t, src, channel, text in self.entries()
        )

    # -- handlers ------------------------------------------------------

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        self.record(message, message.t_created)


async def main() -> None:
    mind = Mind(
        "bicameral",
        SUBJECT,
        system="You are the half of a mind that thinks. You never speak to anyone.",
        ego=EGO,
        ego_system="You are the half of a mind that speaks.",
        run_dir="runs",
    )
    voice = Voice(
        "voice",
        mind.model(VOICE),
        system=(
            "You utter one short sentence about the situation. Never address "
            "the user. Never explain yourself."
        ),
    )
    blackboard = Blackboard()

    mind.intercept(voice)  # prompt -> subject -> voice -> ego -> world
    mind.subject.register(blackboard, "*")
    voice.register(blackboard, "*")

    print(mind.describe(), "\n")

    mind.prompt("What is a digital mind?")
    await mind.process()

    print("\n" + "=" * 70)
    print("spoken:", texts(mind.get_replies())[0])
    print("=" * 70)

    print("\nglobal workspace:")
    print(blackboard.render())

    print(f"\nticks used: {mind.scheduler.t}")

    # The voice and the workspace both took a turn inside tick 1. Prove it.
    turns = [(e.tick, e.module) for e in mind.events.events if e.kind == "handle.start"]
    print(f"turns by tick: {turns}")

    mind.close()


if __name__ == "__main__":
    asyncio.run(main())
