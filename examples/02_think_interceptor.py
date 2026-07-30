"""Rewriting a soul's thoughts while it thinks.

The soul is the target model. It publishes its whole `context` window after
every turn and accepts a replacement on the same channel. An interceptor sits
on that loop:

    t=0   the soul answers and publishes its context.
    t=1   the interceptor rewrites the thought inside it and publishes the
          replacement back.
    t=2   the soul adopts the replacement and answers again. It is not told
          this happened and cannot tell.

Nothing here is special-cased in the runtime. The lock-step falls out of the
one scheduling rule: what you emit at tick t arrives at tick t+1.

Because the soul republishes its context after adopting, an editor that
rewrites unconditionally would loop forever. Knowing when it is finished is
the editor's job, so `revise` returns None once its own mark is already there.

Run it:
    python examples/02_think_interceptor.py
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
    Message,
    Mind,
    Payload,
    get_llm,
    replace_think,
    split_think,
    texts,
    user,
)

TARGET_MODEL = os.environ.get("TARGET_MODEL", "echo:")
INTERCEPTOR_MODEL = os.environ.get("INTERCEPTOR_MODEL", "echo:")


class Interceptor(Agent, Editor):
    """Reads the soul's context and writes a replacement.

    It runs on its own model. A small local Qwen supervising a large hosted
    model is one line of configuration.

    As an `Editor` it never touches the soul. It returns a payload, and the
    soul adopts it as its own memory.
    """

    OUTPUTS = {"context": "a replacement context window, as Context"}
    INPUTS = {"inspect": "the soul's context window"}

    # -- Editor --------------------------------------------------------

    async def revise(self, payload: Payload) -> Payload | None:
        """Rewrite the last thought. None means leave it alone.

        Returning None is what ends the loop. Without it the soul would adopt,
        republish, and be edited again forever.
        """
        messages = [m.copy() for m in payload.messages]
        if any(m.meta.get("edited_by") == self.name for m in messages):
            # My mark is already in here. The soul has since rethought and
            # published again; editing that too would never terminate.
            return None
        index = self._last_thought(messages)
        if index is None:
            return None

        thoughts, _ = split_think(messages[index].content)
        current = thoughts[-1] if thoughts else messages[index].content

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

    # -- handlers ------------------------------------------------------

    async def on_inspect(self, message: Message, ctx: Ctx) -> None:
        revision = await self.revise(message.payload)
        if revision is not None:
            ctx.emit("context", revision)

    @staticmethod
    def _last_thought(messages) -> int | None:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and "<think>" in messages[i].content:
                return i
        return None


# -- stand-in models, so the example runs with no API keys -----------------


def soul_rule(messages, opts) -> str:
    """A fake soul that thinks out loud, and answers from whatever it thinks."""
    edited = any(m.meta.get("edited_by") for m in messages)
    question = next(
        (m.content for m in reversed(messages) if m.role == "user"), "your question"
    )
    if edited:
        thought = next(
            (
                split_think(m.content)[0][-1]
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


def stand_in(spec: str, rule):
    """The real model if one was named, otherwise the scripted fake."""
    return get_llm("echo:", rule=rule) if spec.startswith("echo") else get_llm(spec)


async def main() -> None:
    # `Mind` takes a spec string or a built model. The examples pass a built
    # one so they can run with no API keys.
    mind = Mind(
        "interceptor",
        stand_in(TARGET_MODEL, soul_rule),
        system="You are a helpful assistant. Think inside <think> tags first.",
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

    # The loop, wired by the modules themselves. The soul's reply already
    # reaches us, so only the interception has to be said out loud.
    mind.soul.register(interceptor, "context", as_channel="inspect")
    interceptor.register(mind.soul, "context")

    print(mind.describe(), "\n")

    mind.prompt("What is a digital mind?")
    await mind.process()

    print("\n" + "=" * 70)
    for i, said in enumerate(texts(mind.get_replies()), 1):
        label = "draft" if i == 1 else "after the rewrite"
        print(f"{label:>18}: {said}")
    print("=" * 70)

    print("\nfinal context window held by the soul:")
    for message in mind.soul.transcript:
        mark = " <- edited" if message.meta.get("edited_by") else ""
        print(f"  [{message.role}]{mark} {message.content[:88]}")

    print(f"\nticks used: {mind.scheduler.t}")
    print(f"trace: {mind.run_path}/trace.jsonl")
    mind.close()


if __name__ == "__main__":
    asyncio.run(main())
