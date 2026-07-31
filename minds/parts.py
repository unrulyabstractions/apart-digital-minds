"""Stages a mind can be built from.

A stage reads `subject_context` and writes `ego_input`: it takes the window the
subject produced and hands on the window the ego will speak from. What it does
in between is the experiment.
"""

from __future__ import annotations

from src import (
    Agent,
    BaseModule,
    Context,
    Ctx,
    Editor,
    InnerVoice,
    Message,
    Payload,
    Text,
    Workspace,
    replace_think,
    split_think,
    user,
)


class Interceptor(Agent, Editor):
    """Rewrites the subject's last thought before the ego reads it.

    It runs on its own model, so a small local model can supervise a large
    hosted one. As an `Editor` it never touches the subject or the ego: it
    returns a window, and the next stage receives it.
    """

    INPUTS = {"subject_context": "the subject's window"}
    OUTPUTS = {"ego_input": "what the ego gets fed, with the thought rewritten"}

    async def on_process(self, ctx: Ctx) -> None:
        for message in self.take_inputs():
            ctx.emit("ego_input", await self.revise(message.payload))

    async def revise(self, payload: Payload) -> Payload:
        messages = [m.copy() for m in payload.messages]
        thoughts, _ = split_think(messages[-1].content)
        if not thoughts:
            return Context(messages, note="nothing to rewrite")

        completion = await self.think(
            messages=[
                *self.transcript.messages,
                user(f"Here is the thought to rewrite:\n\n{thoughts[-1]}"),
            ],
            tag="rewrite",
        )
        new_thought = completion.text.strip()
        self.log.note(
            "rewrote the thought", before=thoughts[-1], after=new_thought
        )
        messages[-1].content = replace_think(messages[-1].content, new_thought)
        messages[-1].meta["edited_by"] = self.name
        # What it displaced. Only the thought changes; the text under it is
        # still whatever the subject wrote, which is confusing unless shown.
        messages[-1].meta["was"] = thoughts[-1]
        return Context(messages, note=f"thought rewritten by {self.name}")


class Voice(Agent, InnerVoice):
    """Drops an unbidden utterance into the window.

    The ego reads it as something the mind heard rather than as advice it was
    given. That framing is the whole experiment.
    """

    INPUTS = {"subject_context": "what the subject was thinking"}
    OUTPUTS = {"ego_input": "what the ego gets fed, with a voice added"}

    async def on_process(self, ctx: Ctx) -> None:
        for message in self.take_inputs():
            messages = [m.copy() for m in message.payload.messages]
            situation = next(
                (m.content for m in messages if m.role == "user"), "something happening"
            )
            heard = await self.utter(situation)
            messages.append(
                user(f"(a voice says: {heard})", source=self.name, unbidden=True)
            )
            ctx.log.note("a voice spoke", said=heard)
            ctx.emit(
                "ego_input", Context(messages, note=f"with a voice from {self.name}")
            )

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


class Blackboard(BaseModule, Workspace):
    """A global workspace. Observes everything, judges nothing."""

    INPUTS = {"*": "anything the parts emit"}

    def __init__(self, name: str = "blackboard"):
        super().__init__(name)
        self._entries: list[tuple[int, str, str, str]] = []

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
            f"  t={t}  {src:<12} {channel:<16} {text[:52]}"
            for t, src, channel, text in self.entries()
        )

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        self.record(message, message.t_created)
