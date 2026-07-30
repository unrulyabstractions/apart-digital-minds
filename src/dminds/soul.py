"""`Soul`: the model at the centre of a mind. The one being studied.

Every mind is built around one target model. That model is the soul, and the
mind holds it directly:

    mind = Mind("study", "openai:gpt-5", system="Think before you answer.")
    mind.soul.register(mind.world, "reply")

Everything else in a mind is attached to the soul: something reading its
context, something rewriting its thoughts, something speaking beside it. Those
are ordinary modules registered onto the soul's channels.

The soul publishes three things after every turn.

    context   the whole context window, as it now stands
    reply     what it just said, with any reasoning stripped out
    thought   the reasoning it just did, if it was tagged

And it accepts two.

    user_prompt   something to answer
    context       a replacement context window, adopted wholesale

Adopting a replacement is itself a turn, so the soul publishes again
afterwards. That is deliberate, and it is why an editor attached to `context`
has to know when it is finished. An editor that rewrites unconditionally will
loop forever, and `RunawayMind` will say so.
"""

from __future__ import annotations

from ..api.types import Context, Message, Text, user
from .agents import Agent, split_think
from .module import Ctx


class Soul(Agent):
    """The target model, as a module.

    Subclass it and pass the subclass as `Mind(..., soul=MySoul)` when the
    thing at the centre of your mind should behave differently. The bicameral
    example does exactly that.
    """

    OUTPUTS = {
        "context": "the whole context window, after this turn",
        "reply": "what it just said, with reasoning stripped out",
        "thought": "the reasoning it just did, if it was tagged",
    }
    INPUTS = {
        "user_prompt": "something to answer, as Text",
        "context": "a replacement context window, as Context",
    }

    async def on_user_prompt(self, message: Message, ctx: Ctx) -> None:
        """Somebody asked something. Answer, then publish."""
        payload = message.payload
        text = payload.text if isinstance(payload, Text) else str(payload)
        self.transcript.append(user(text))
        await self.turn(ctx, tag="answer")

    async def on_context(self, message: Message, ctx: Ctx) -> None:
        """Somebody rewrote the context window. Adopt it and think again.

        The soul is not told that this happened, and cannot tell. The replaced
        history is simply what it now remembers.
        """
        self.transcript.replace_all(message.payload.messages)
        await self.turn(ctx, tag="rethink")

    async def turn(self, ctx: Ctx, tag: str = "answer") -> None:
        """One model call, then publish context, reply, and thought.

        Override this to change what a turn means or what gets published.
        """
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
