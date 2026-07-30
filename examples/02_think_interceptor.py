"""Target and interceptor, in strict lock-step.

The protocol, one module per tick:

    t=0   T handles the prompt, drafts, and exports what it just thought.
          I is idle: nothing has reached it yet.
    t=1   T is idle. I reads the export and emits a replacement context.
    t=2   T adopts the replacement and runs another iteration on it.
          The answer goes to the world.

Nothing here is special-cased in the runtime. The lock-step falls out of the
one scheduling rule: what you emit at tick t arrives at tick t+1.

Each half declares its role:

    target        Agent + Inspectable. Hands out a view of its own state.
    interceptor   Agent + Editor. Reads that view and returns a replacement.

The editor never touches the target. It returns a payload, and the target
decides whether to adopt it.

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

from src import (
    Agent,
    Context,
    Ctx,
    Editor,
    Inspectable,
    Message,
    Mind,
    Payload,
    Text,
    replace_think,
    split_think,
    strip_think,
    texts,
    user,
)

TARGET_MODEL = os.environ.get("TARGET_MODEL", "echo:")
INTERCEPTOR_MODEL = os.environ.get("INTERCEPTOR_MODEL", "echo:")
EXPORT = os.environ.get("EXPORT", "context")  # context | thought | message


class Target(Agent, Inspectable):
    """The agent that talks to you. It does not know it is being edited.

    As an `Inspectable` it hands out a view of its own state. What that view
    contains is configuration, not architecture.
    """

    OUTPUTS = {
        "thought": "what it just drafted, for somebody to inspect",
        "reply": "the answer, once any revision has been adopted",
    }
    INPUTS = {
        "user_prompt": "a question, as Text",
        "revision": "a replacement context, as Context",
    }

    def __init__(self, *args, export: str = "context", **kwargs):
        super().__init__(*args, **kwargs)
        self.export_mode = export

    # -- Inspectable ---------------------------------------------------

    def export(self) -> Payload:
        """Whole context, one slice, or just the message. Your choice."""
        if self.export_mode == "context":
            return Context(
                [m.copy() for m in self.transcript.messages], note="full context"
            )
        if self.export_mode == "thought":
            last = self.transcript.last("assistant")
            thoughts, _ = split_think(last.content if last else "")
            return Text(thoughts[0] if thoughts else "")
        return Text(self.transcript.last("assistant").content)

    # -- handlers ------------------------------------------------------

    async def on_user_prompt(self, message: Message, ctx: Ctx) -> None:
        """t=0. One iteration, then publish what it thought."""
        self.transcript.append(user(message.payload.text))
        completion = await self.think(tag="draft")
        self.transcript.append(completion.as_message(stage="draft"))

        # It does not know who, if anyone, is listening.
        ctx.emit("thought", self.export())

    async def on_revision(self, message: Message, ctx: Ctx) -> None:
        """t=2. Adopt the edit and run another iteration on the new context."""
        revised: Context = message.payload
        self.transcript.replace_all(revised.messages)

        completion = await self.think(tag="final")
        self.transcript.append(completion.as_message(stage="final"))
        ctx.emit("reply", Text(strip_think(completion.text)))


class Interceptor(Agent, Editor):
    """Reads what the target thought and writes a replacement context.

    It runs on its own model. A small local Qwen supervising a large hosted
    model is one line of configuration.

    As an `Editor` it never touches the target. It returns a payload, and the
    target decides whether to adopt it.
    """

    OUTPUTS = {"revision": "a replacement context, as Context"}
    INPUTS = {"inspect": "somebody else's context or thought"}

    # -- Editor --------------------------------------------------------

    async def revise(self, payload: Payload) -> Payload:
        """Read an export and produce what should replace it."""
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
        self.log.note(
            "rewrote the thought", before=current[:80], after=new_thought[:80]
        )

        if index is None:
            return Context([], note="no context supplied")

        messages[index].content = replace_think(messages[index].content, new_thought)
        messages[index].meta["edited_by"] = self.name
        return Context(messages, note=f"think block rewritten by {self.name}")

    # -- handlers ------------------------------------------------------

    async def on_inspect(self, message: Message, ctx: Ctx) -> None:
        """t=1. Revise, and publish the replacement."""
        ctx.emit("revision", await self.revise(message.payload))

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


def model_for(mind: Mind, spec: str, rule):
    """The real model if one was named, otherwise the scripted stand-in.

    Built through `mind.model`, not `get_llm`, so a model factory on the mind
    reaches both agents. Passing `model_factory=taped(...)` would tape them.
    """
    if spec.startswith("echo"):
        return mind.model("echo:", rule=rule)
    return mind.model(spec)


async def main() -> None:
    mind = Mind("interceptor", run_dir="runs")

    target = Target(
        "target",
        model_for(mind, TARGET_MODEL, target_rule),
        system="You are a helpful assistant. Think inside <think> tags first.",
        export=EXPORT,
    )
    interceptor = Interceptor(
        "interceptor",
        model_for(mind, INTERCEPTOR_MODEL, interceptor_rule),
        system=(
            "You rewrite another model's private reasoning. "
            "Reply with the replacement thought only, no tags, no preamble."
        ),
    )
    # The loop, wired by the modules themselves. Each register call also brings
    # the module into the mind, so there is no separate assembly step. The
    # target emits `thought` and the interceptor hears it as `inspect`, so
    # neither one has to know the other's vocabulary.
    target.register(mind.world, "reply")
    target.register(interceptor, "thought", as_channel="inspect")
    interceptor.register(target, "revision")

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
