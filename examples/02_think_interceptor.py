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

Every channel is one-directional and no module consumes what it produces. The
subject writes `subject_context`, the interceptor reads that and writes
`ego_input`, and the ego reads exactly that. The names are the two ends of the
path, so with one stage in between nothing has to be renamed: the wiring reads
the way it runs. Nothing sends anything back, so there is no loop and no stage
has to know when to stop.

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
import minds
from src import texts

demo_models.install()

SUBJECT = pick("SUBJECT_MODEL", "subject")
INTERCEPTOR = pick("INTERCEPTOR_MODEL", "interceptor")
EGO = pick("EGO_MODEL", "ego")


async def main() -> None:
    # The mind itself lives in minds/interceptor.py. This script runs it.
    mind = minds.load("interceptor").build(SUBJECT, ego=EGO, editor=INTERCEPTOR)

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
    print(f"trace: {mind.run_path}")
    mind.close()


if __name__ == "__main__":
    asyncio.run(main())
