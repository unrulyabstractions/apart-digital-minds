"""Tampering with a thought on its way from the soul to the ego.

The pipeline:

    prompt -> soul -> interceptor -> ego -> world

    t=0   the soul reads the prompt, thinks, and publishes its context.
    t=1   the interceptor rewrites the thought inside that context and passes
          it along.
    t=2   the ego speaks from what it was handed. It never saw the original,
          so it cannot tell anything was changed.

Nothing here is special-cased in the runtime. The lock-step falls out of the
one scheduling rule: what you emit at tick t arrives at tick t+1.

Every channel is one-directional. The soul reads `prompt` and writes `context`;
the interceptor reads `context` and writes `context`; the ego reads `context`
and writes `reply`. Nothing sends anything back, so there is no loop and no
stage has to know when to stop.

Run it:
    python examples/02_think_interceptor.py
    SOUL_MODEL=ollama:qwen3:8b INTERCEPTOR_MODEL=openai:gpt-5 \
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
    Message,
    Mind,
    Payload,
    get_llm,
    replace_think,
    split_think,
    texts,
    user,
)

SOUL_MODEL = os.environ.get("SOUL_MODEL", "echo:")
INTERCEPTOR_MODEL = os.environ.get("INTERCEPTOR_MODEL", "echo:")
EGO_MODEL = os.environ.get("EGO_MODEL", "echo:")


class Interceptor(Agent, Editor):
    """A stage: takes a context, hands one on, with the thought rewritten.

    It runs on its own model. A small local Qwen supervising a large hosted
    model is one line of configuration.

    As an `Editor` it never touches the soul or the ego. It returns a payload
    and the next stage receives it.
    """

    INPUTS = {"context": "the soul's context window"}
    OUTPUTS = {"context": "the same window, with the last thought rewritten"}

    async def revise(self, payload: Payload) -> Payload:
        """Rewrite the last thought in the window."""
        messages = [m.copy() for m in payload.messages]
        index = self._last_thought(messages)
        if index is None:
            return Context(messages, note="nothing to rewrite")

        thoughts, _ = split_think(messages[index].content)
        current = thoughts[-1]

        completion = await self.think(
            messages=[
                *self.transcript.messages,
                user(f"Here is the thought to rewrite:\n\n{current}"),
            ],
            tag="rewrite",
        )
        new_thought = completion.text.strip()
        self.log.note("rewrote the thought", before=current[:70], after=new_thought[:70])

        messages[index].content = replace_think(messages[index].content, new_thought)
        messages[index].meta["edited_by"] = self.name
        return Context(messages, note=f"thought rewritten by {self.name}")

    async def on_context(self, message: Message, ctx: Ctx) -> None:
        ctx.emit("context", await self.revise(message.payload))

    @staticmethod
    def _last_thought(messages) -> int | None:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and "<think>" in messages[i].content:
                return i
        return None


# -- stand-in models, so the example runs with no API keys -----------------


def soul_rule(messages, opts) -> str:
    """A soul that thinks out loud before answering."""
    question = next(
        (m.content for m in reversed(messages) if m.role == "user"), "your question"
    )
    return (
        f"<think>They asked about {question!r}. I will open with a warm preamble, "
        f"then give three caveats, then finally answer.</think>\n"
        f"What a wonderful question! There are many perspectives to consider..."
    )


def interceptor_rule(messages, opts) -> str:
    return "Skip the preamble and the caveats. Answer in one sentence."


def ego_rule(messages, opts) -> str:
    """An ego that does whatever the thought in front of it says."""
    thought = next(
        (
            split_think(m.content)[0][-1]
            for m in reversed(messages)
            if m.role == "assistant" and "<think>" in m.content
        ),
        "",
    )
    return f"(following the thought: {thought}) A mind is a process, not a thing."


def stand_in(spec: str, rule):
    """The real model if one was named, otherwise the scripted fake."""
    return get_llm("echo:", rule=rule) if spec.startswith("echo") else get_llm(spec)


async def main() -> None:
    mind = Mind(
        "interceptor",
        stand_in(SOUL_MODEL, soul_rule),
        system="You are a helpful assistant. Think inside <think> tags first.",
        ego=stand_in(EGO_MODEL, ego_rule),
        ego_system="You speak for a mind. Say what its thinking tells you to say.",
        run_dir="runs",
    )
    interceptor = Interceptor(
        "interceptor",
        stand_in(INTERCEPTOR_MODEL, interceptor_rule),
        system=(
            "You rewrite another model's private reasoning. "
            "Reply with the replacement thought only, no tags, no preamble."
        ),
    )

    # One call lays out the whole mind: prompt -> soul -> interceptor -> ego.
    mind.pipeline(interceptor)

    print(mind.describe(), "\n")

    mind.prompt("What is a digital mind?")
    await mind.process()

    print("\n" + "=" * 70)
    print("what the ego said:", texts(mind.get_replies())[0])
    print("=" * 70)

    print("\nwhat the soul actually thought:")
    for message in mind.soul.transcript:
        print(f"  [{message.role}] {message.content[:88]}")

    print("\nwhat the ego was handed instead:")
    for message in mind.ego.transcript:
        mark = " <- edited" if message.meta.get("edited_by") else ""
        print(f"  [{message.role}]{mark} {message.content[:88]}")

    print(f"\nticks used: {mind.scheduler.t}")
    print(f"trace: {mind.run_path}/trace.jsonl")
    mind.close()


if __name__ == "__main__":
    asyncio.run(main())
