"""`Agent`: a module with a model, a transcript, and instrumented calls.

This is deliberately thin. It gives you one instrumented way to call a model
and one obvious default handler, and then gets out of the way. Every
architecture in `examples/` is built by subclassing this and writing `on_*`
methods. None of them is baked in here.

    class Assistant(Agent):
        async def on_user_prompt(self, task, ctx):
            self.transcript.append(user(task.payload.text))
            completion = await self.think(tag="answer")
            self.transcript.append(completion.as_message())
            ctx.emit("reply", Text(completion.text), to="world")
"""

from __future__ import annotations

import re
from typing import Sequence

from ..api.models import LLM
from ..api.observability import LLM_ERROR, LLM_REQUEST, LLM_RESPONSE
from ..api.types import (
    ChatMessage,
    Completion,
    GenOptions,
    Task,
    Text,
    system as system_msg,
    user,
)
from .memory import Journal, Scratchpad, Transcript
from .module import BaseModule, Ctx


class Agent(BaseModule):
    """A module backed by a chat model.

    Args:
        name: how it appears in routes and logs.
        llm: any `LLM`. Swap providers by changing the spec string.
        system: the system prompt, seeded as the first transcript message.
        opts: default sampling options for this agent's calls.
        reply_to: where the default handler sends its answer. `"world"` hands
            it back to the caller. Set to `None` to use the wired routes.
        max_context: if set, only the last N messages are sent to the model.
            The transcript keeps everything either way.
    """

    def __init__(
        self,
        name: str,
        llm: LLM,
        system: str | None = None,
        opts: GenOptions | None = None,
        reply_to: str | None = "world",
        max_context: int | None = None,
        journal: Journal | None = None,
    ):
        super().__init__(name)
        self.llm = llm
        self.opts = opts or GenOptions()
        self.reply_to = reply_to
        self.max_context = max_context
        self.transcript = Transcript()
        self.scratch = Scratchpad()
        self.journal = journal
        if system:
            self.transcript.messages.append(system_msg(system))

    def attach(self, mind) -> None:
        super().attach(mind)
        # Wire the logger through so memory writes appear in this module's file.
        self.transcript.log = self.log
        self.scratch.log = self.log
        if self.journal is not None and self.journal.log is None:
            self.journal.log = self.log

    # -- calling the model ----------------------------------------------

    def prompt_messages(self) -> list[ChatMessage]:
        """What gets sent to the model. Override to change context assembly."""
        if self.max_context is None:
            return list(self.transcript.messages)
        return self.transcript.window(self.max_context)

    async def think(
        self,
        messages: Sequence[ChatMessage] | None = None,
        opts: GenOptions | None = None,
        tag: str = "",
    ) -> Completion:
        """One model call, fully logged.

        Pass `messages` to call on something other than this agent's transcript,
        which is how a monitor reasons about another agent's context.
        """
        msgs = list(messages) if messages is not None else self.prompt_messages()
        opts = opts or self.opts
        last = msgs[-1].content if msgs else ""

        self.log.event(
            LLM_REQUEST,
            model=self.llm.spec,
            tag=tag,
            n_messages=len(msgs),
            temperature=opts.temperature,
            max_tokens=opts.max_tokens,
            summary=f"-> {self.llm.spec} [{tag or 'call'}] {len(msgs)} msgs: {last[:60]}",
        )
        try:
            completion = await self.llm.chat(msgs, opts)
        except Exception as exc:
            self.log.event(
                LLM_ERROR,
                model=self.llm.spec,
                tag=tag,
                error=repr(exc),
                summary=f"!! {self.llm.spec} failed: {exc!r}",
            )
            raise

        self.log.event(
            LLM_RESPONSE,
            model=completion.model,
            tag=tag,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            latency_s=round(completion.latency_s, 4),
            text=completion.text,
            summary=f"<- {self.llm.spec} [{tag or 'call'}] {completion.text[:60]}",
        )
        return completion

    async def say(self, text: str, tag: str = "say") -> Completion:
        """Append a user turn, call the model, append the reply. The common case."""
        self.transcript.append(user(text))
        completion = await self.think(tag=tag)
        self.transcript.append(completion.as_message(tag=tag))
        return completion

    # -- default handler ---------------------------------------------------

    async def on_user_prompt(self, task: Task, ctx: Ctx) -> None:
        """Answer a prompt and send the answer to `reply_to`."""
        text = task.payload.text if isinstance(task.payload, Text) else str(task.payload)
        completion = await self.say(text, tag="reply")
        ctx.emit("reply", Text(completion.text), to=self.reply_to)

    def close(self) -> None:
        self.llm.close()


# -- working with <think> blocks -----------------------------------------
#
# Generic string helpers. They assume nothing about who writes the tags or what
# a rewrite means, so they compose with any architecture you build.

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_think(text: str) -> tuple[list[str], str]:
    """Separate thinking from output.

    Returns every `<think>` block and the text with those blocks removed.
    """
    thoughts = [m.group(1).strip() for m in THINK_RE.finditer(text)]
    return thoughts, THINK_RE.sub("", text).strip()


def strip_think(text: str) -> str:
    """Just the visible part."""
    return THINK_RE.sub("", text).strip()


def replace_think(text: str, new_thought: str, index: int = 0) -> str:
    """Swap the contents of one `<think>` block, keeping everything else.

    If the text has no think block, one is prepended.
    """
    matches = list(THINK_RE.finditer(text))
    if not matches:
        return f"<think>{new_thought}</think>\n{text}"
    if index >= len(matches):
        index = len(matches) - 1
    match = matches[index]
    return text[: match.start()] + f"<think>{new_thought}</think>" + text[match.end() :]


def has_think(text: str) -> bool:
    return THINK_RE.search(text) is not None
