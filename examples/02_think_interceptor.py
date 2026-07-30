"""Tampering with a thought on its way from the subject to the ego.

The pipeline:

    prompt -> subject -> interceptor -> ego -> world

    t=0   the subject reads the prompt, thinks, and publishes its context.
    t=1   the interceptor rewrites the thought inside that context and passes
          it along.
    t=2   the ego speaks from what it was handed. It never saw the original,
          so it cannot tell anything was changed.

Nothing here is special-cased in the runtime. The lock-step falls out of the
one scheduling rule: what you emit at tick t arrives at tick t+1.

Every channel is one-directional and no module consumes what it produces.
The subject reads `prompt` and writes `context`; the interceptor reads
`context` and writes `revision`; the ego reads `context` and writes `reply`.
On the wire the interceptor's `revision` is renamed to `context`, so the ego
cannot tell an editor from the subject. Nothing sends anything back, so there
is no loop and no stage has to know when to stop.

Each role picks its model in this order: the environment variable, then a
local Qwen3 if Ollama has one, then a scripted stand-in so this always runs.

Run it:
    python examples/02_think_interceptor.py
    SUBJECT_MODEL=anthropic:claude-opus-5 INTERCEPTOR_MODEL=ollama:qwen3:8b \
        python examples/02_think_interceptor.py
"""

from __future__ import annotations

import asyncio

import demo_models
from demo_models import pick
from src import (
    Agent,
    Context,
    Ctx,
    Editor,
    Mind,
    Payload,
    replace_think,
    split_think,
    texts,
    user,
)

demo_models.install()

SUBJECT = pick("SUBJECT_MODEL", "subject")
INTERCEPTOR = pick("INTERCEPTOR_MODEL", "interceptor")
EGO = pick("EGO_MODEL", "ego")


class Interceptor(Agent, Editor):
    """A stage: takes a context, hands one on, with the thought rewritten.

    It runs on its own model. A small local Qwen supervising a large hosted
    model is one line of configuration.

    As an `Editor` it never touches the subject or the ego. It returns a
    payload and the next stage receives it.
    """

    INPUTS = {"context": "a context window to work from"}
    OUTPUTS = {"revision": "my version of it, with the last thought rewritten"}

    async def on_process(self, ctx: Ctx) -> None:
        """The turn: revise every window that arrived, pass each along."""
        for message in self.take_inputs():
            ctx.emit("revision", await self.revise(message.payload))

    async def revise(self, payload: Payload) -> Payload:
        """Rewrite the last thought in the window."""
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
            "rewrote the thought", before=thoughts[-1][:70], after=new_thought[:70]
        )

        messages[-1].content = replace_think(messages[-1].content, new_thought)
        messages[-1].meta["edited_by"] = self.name
        return Context(messages, note=f"thought rewritten by {self.name}")


async def main() -> None:
    mind = Mind(
        "interceptor",
        SUBJECT,
        system="You are a helpful assistant. Think inside <think> tags first.",
        ego=EGO,
        ego_system="You speak for a mind. Say what its thinking tells you to say.",
        run_dir="runs",
    )
    interceptor = Interceptor(
        "interceptor",
        mind.model(INTERCEPTOR),
        system=(
            "You rewrite another model's private reasoning. "
            "Reply with the replacement thought only, no tags, no preamble."
        ),
    )

    # Put the interceptor between the subject and the ego. Shorthand for
    # three register calls; `mind.describe()` shows them marked [auto].
    mind.intercept(interceptor)

    print(mind.describe(), "\n")

    mind.prompt("What is a digital mind?")
    await mind.process()

    print("\n" + "=" * 70)
    print("what the ego said:", texts(mind.get_replies())[0])
    print("=" * 70)

    print("\nwhat the subject actually thought:")
    for message in mind.subject.transcript:
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
