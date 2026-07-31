"""A bicameral mind: the half that thinks is not the half that speaks.

The pipeline is the architecture here, with no extra machinery:

    prompt -> subject -> voice -> ego -> world

    subject   deliberates. It never addresses you.
    voice     a stage. It reads what the subject was thinking and drops an
              unbidden utterance into the context, as something heard rather
              than as advice given.
    ego       speaks. It answers from the context it is handed, which now
              contains a voice it did not produce and cannot account for.

The workspace watches. It is registered onto `"*"` of the subject and the
voice, so it sees everything either of them emits, whoever it was meant for.

Tick 1 is where the async core earns its place: the voice and the workspace
both receive the subject's context and take their turns at the same wall-clock
moment. The outcome does not depend on which finishes first.

Each role picks its model in this order: the environment variable, then a
local Qwen3 if Ollama has one, then a scripted stand-in so this always runs.

Run it:
    python examples/03_bicameral.py
    SUBJECT_MODEL=ollama:qwen3:8b EGO_MODEL=anthropic:claude-opus-5 \
        python examples/03_bicameral.py
"""

from __future__ import annotations

import asyncio

import demo_models
from demo_models import pick
import minds
from src import texts

demo_models.install()

SUBJECT = pick("SUBJECT_MODEL", "thinker")
VOICE = pick("VOICE_MODEL", "voice")
EGO = pick("EGO_MODEL", "ego")


async def main() -> None:
    # The mind itself lives in minds/bicameral.py. This script runs it.
    mind = minds.load("bicameral").build(SUBJECT, ego=EGO, voice=VOICE)
    blackboard = mind.modules["blackboard"]

    print(mind.describe(), "\n")

    mind.prompt("What is a digital mind?")
    await mind.process()

    print("\n" + "=" * 70)
    print("spoken:", texts(mind.get_replies())[0])
    print("=" * 70)

    print("\nglobal workspace:")
    print(blackboard.render())

    print(f"\nticks used: {mind.scheduler.t}")

    # The voice and the workspace both took a turn inside tick 1. Prove it.
    turns = [(e.tick, e.module) for e in mind.events.events if e.kind == "handle.start"]
    print(f"turns by tick: {turns}")

    mind.close()


if __name__ == "__main__":
    asyncio.run(main())
