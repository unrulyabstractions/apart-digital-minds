"""`Subject` and `Ego`: the two named parts a mind can be built around.

A mind is a pipeline with a fixed shape:

    prompt -> subject -> [stages] -> ego -> world

    subject   the target model. It reads the prompt, thinks, and publishes
              what it thought. It is the thing an experiment is about.
    stages    anything you put in between. Each one takes a context and passes
              a context along. This is where interception lives.
    ego       the part that speaks. It is given a context, possibly an edited
              one, and produces the reply you see.

The ego is optional. Without one the subject's own reply goes straight to you,
so the simplest mind is a subject and nothing else.

Every channel is one-directional and means one thing. No module consumes what
it produces, so the path cannot loop and no stage has to know when to stop.

    subject   reads  prompt     writes  context, reply, thought
    stage     reads  context    writes  revised
    ego       reads  context    writes  reply

A stage's `revised` is renamed to `context` on the wire, so the next hop
cannot tell an editor from the subject. The names differ at the module so
that no module consumes what it produces.
"""

from __future__ import annotations

from ..api.types import Context, Message, Text, user
from .agents import Agent, split_think
from .module import Ctx


class Subject(Agent):
    """The target model, as a module.

    Reads a prompt, takes one turn, and publishes three separate things. What
    happens to them afterwards is not its concern, and it has no way to find
    out.

    Subclass it and pass the subclass as `Mind(..., subject=MySubject)`.
    """

    INPUTS = {"prompt": "something to answer, as Text"}
    OUTPUTS = {
        "context": "the whole context window after this turn, as Context",
        "reply": "what it said, with reasoning stripped out, as Text",
        "thought": "the reasoning it did, if it was tagged, as Text",
    }

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        """A prompt arrived. It becomes a user turn in the window."""
        payload = message.payload
        text = payload.text if isinstance(payload, Text) else str(payload)
        self.transcript.append(user(text))

    async def on_process(self, ctx: Ctx) -> None:
        """One turn over everything that arrived."""
        await self.turn(ctx)

    async def turn(self, ctx: Ctx, tag: str = "answer") -> None:
        """One model call, then publish. Override to change what a turn means."""
        completion = await self.think(tag=tag)
        self.transcript.append(completion.as_message(stage=tag))

        thoughts, visible = split_think(completion.text)
        if thoughts:
            ctx.emit("thought", Text(thoughts[-1]))
        ctx.emit("reply", Text(visible))
        ctx.emit(
            "context",
            Context([m.copy() for m in self.transcript.messages], note=f"after {tag}"),
        )


class Ego(Agent):
    """The part of a mind that speaks.

    It is handed a context window, which may have been edited on the way, and
    answers from it. It never sees the original, so it cannot tell that
    anything was changed.

    Give a mind an ego and the reply you receive comes from here instead of
    from the subject.
    """

    INPUTS = {"context": "a context window to speak from, as Context"}
    OUTPUTS = {"reply": "what it says, as Text"}

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        """A context window arrived. It becomes this module's memory."""
        self.transcript.replace_all(message.payload.messages)

    async def on_process(self, ctx: Ctx) -> None:
        """Speak from whatever window it now holds."""
        completion = await self.think(tag="speak")
        self.transcript.append(completion.as_message(stage="spoken"))
        ctx.emit("reply", Text(split_think(completion.text)[1]))
