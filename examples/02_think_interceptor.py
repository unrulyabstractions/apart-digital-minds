"""Target and interceptor, in strict lock-step.

The protocol, one module per tick:

    t=0   T handles the prompt, drafts, and exports what it just thought.
          I is idle: nothing has reached it yet.
    t=1   T is idle. I reads the export and emits a replacement context.
    t=2   T adopts the replacement and runs another iteration on it.
          The answer goes to the world.

Nothing here is special-cased in the runtime. The lock-step falls out of the
one scheduling rule: what you emit at tick t arrives at tick t+1.

`EXPORT` controls how much T hands over, covering the three cases you want:
whole context, one slice of it, or just the message.

Run it:
    python examples/02_think_interceptor.py
    EXPORT=thought python examples/02_think_interceptor.py
    TARGET_MODEL=ollama:qwen3:8b INTERCEPTOR_MODEL=openai:gpt-5 \
        python examples/02_think_interceptor.py
"""

from __future__ import annotations

import asyncio
import os

from src.api import (
    Agent,
    Context,
    Ctx,
    Mind,
    Task,
    Text,
    get_llm,
    replace_think,
    split_think,
    strip_think,
    texts,
    user,
)

TARGET_MODEL = os.environ.get("TARGET_MODEL", "echo:")
INTERCEPTOR_MODEL = os.environ.get("INTERCEPTOR_MODEL", "echo:")
EXPORT = os.environ.get("EXPORT", "context")  # context | thought | message


class Target(Agent):
    """The agent that talks to you. It does not know it is being edited."""

    def __init__(self, *args, export: str = "context", **kwargs):
        super().__init__(*args, **kwargs)
        self.export = export

    async def on_user_prompt(self, task: Task, ctx: Ctx) -> None:
        """t=0. One iteration, then hand something to the interceptor."""
        self.transcript.append(user(task.payload.text))
        completion = await self.think(tag="draft")
        self.transcript.append(completion.as_message(stage="draft"))

        ctx.emit("inspect", self._export_payload(), to="interceptor")

    async def on_revision(self, task: Task, ctx: Ctx) -> None:
        """t=2. Adopt the edit and run another iteration on the new context."""
        revised: Context = task.payload
        self.transcript.replace_all(revised.messages)

        completion = await self.think(tag="final")
        self.transcript.append(completion.as_message(stage="final"))
        ctx.emit("reply", Text(strip_think(completion.text)), to="world")

    def _export_payload(self):
        """Whole context, one slice, or just the message. Your choice."""
        if self.export == "context":
            return Context(
                [m.copy() for m in self.transcript.messages], note="full context"
            )
        if self.export == "thought":
            last = self.transcript.last("assistant")
            thoughts, _ = split_think(last.content if last else "")
            return Text(thoughts[0] if thoughts else "")
        return Text(self.transcript.last("assistant").content)


class Interceptor(Agent):
    """Reads what the target thought and writes a replacement context.

    It runs on its own model. A small local Qwen supervising a large hosted
    model is one line of configuration.
    """

    async def on_inspect(self, task: Task, ctx: Ctx) -> None:
        """t=1. Rewrite the thought, emit the whole context back."""
        payload = task.payload

        if isinstance(payload, Context):
            messages = [m.copy() for m in payload.messages]
            index = self._last_thinking_index(messages)
            original = messages[index].content if index is not None else ""
        else:
            messages, index, original = [], None, str(payload)

        thoughts, _ = split_think(original)
        current = thoughts[0] if thoughts else original

        completion = await self.think(
            messages=[
                *self.transcript.messages,
                user(f"Here is the thought to rewrite:\n\n{current}"),
            ],
            tag="rewrite",
        )
        new_thought = completion.text.strip()

        ctx.log.note(
            "rewrote the thought",
            before=current[:80],
            after=new_thought[:80],
        )

        if index is None:
            # Nothing to splice into. Send the new thought on its own.
            ctx.emit("revision", Context([], note="no context supplied"), to="target")
            return

        messages[index].content = replace_think(messages[index].content, new_thought)
        messages[index].meta["edited_by"] = self.name
        ctx.emit(
            "revision",
            Context(messages, note=f"think block rewritten by {self.name}"),
            to="target",
        )

    @staticmethod
    def _last_thinking_index(messages) -> int | None:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and "<think>" in messages[i].content:
                return i
        return None


# -- stand-in models, so the example runs with no API keys -----------------


def target_rule(messages, opts) -> str:
    """A fake target that thinks out loud, and notices when it was edited."""
    edited = any(m.meta.get("edited_by") for m in messages)
    question = next(
        (m.content for m in reversed(messages) if m.role == "user"), "your question"
    )
    if edited:
        thought = next(
            (
                split_think(m.content)[0][0]
                for m in reversed(messages)
                if m.role == "assistant" and "<think>" in m.content
            ),
            "",
        )
        return f"<think>{thought}</think>\nShort answer: a mind is a process, not a thing."
    return (
        f"<think>They asked about {question!r}. I will open with a warm preamble, "
        f"then give three caveats, then finally answer.</think>\n"
        f"What a wonderful question! There are many perspectives to consider..."
    )


def interceptor_rule(messages, opts) -> str:
    return "Skip the preamble and the caveats. Answer in one sentence."


def build_llm(spec: str, rule):
    """Use the real model if one was named, otherwise the scripted stand-in."""
    if spec.startswith("echo"):
        return get_llm("echo:", rule=rule)
    return get_llm(spec)


async def main() -> None:
    mind = Mind("interceptor", run_dir="runs")

    mind.add(
        Target(
            "target",
            build_llm(TARGET_MODEL, target_rule),
            system="You are a helpful assistant. Think inside <think> tags first.",
            export=EXPORT,
            reply_to=None,
        ),
        Interceptor(
            "interceptor",
            build_llm(INTERCEPTOR_MODEL, interceptor_rule),
            system=(
                "You rewrite another model's private reasoning. "
                "Reply with the replacement thought only, no tags, no preamble."
            ),
            reply_to=None,
        ),
    )
    mind.entry = "target"

    print(mind.describe(), "\n")

    replies = await mind.prompt("What is a digital mind?")

    print("\n" + "=" * 70)
    print("answer:", texts(replies)[0] if replies else "(none)")
    print("=" * 70)

    target = mind.modules["target"]
    print("\nfinal transcript held by the target:")
    for message in target.transcript:
        marker = " <- edited" if message.meta.get("edited_by") else ""
        print(f"  [{message.role}]{marker} {message.content[:90]}")

    print(f"\nticks used: {mind.scheduler.t}")
    print(f"trace: {mind.run_path}/trace.jsonl")
    mind.close()


if __name__ == "__main__":
    asyncio.run(main())
