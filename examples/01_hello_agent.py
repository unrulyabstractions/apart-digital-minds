"""One mind, one subject, and a full trace on disk.

A mind is built around one target model. That model is its **subject**, and it
publishes three channels: `context`, `reply`, and `thought`. Its `reply` is
connected to you already, so this example wires nothing. Register consumers
onto `context` or `thought` when you want to read or edit what it remembers.

Driving a mind is three steps, kept apart on purpose:

    mind.prompt(...)      put something in
    await mind.process()  let it think until nothing has work
    mind.get_replies()    read what came out

The model is chosen in this order: `MODEL`, then a local Qwen3 if Ollama has
one, then a scripted stand-in so this always runs.

Run it:
    python examples/01_hello_agent.py

Swap the model without touching anything else:
    MODEL=openai:gpt-5              python examples/01_hello_agent.py
    MODEL=anthropic:claude-opus-5   python examples/01_hello_agent.py
    MODEL=gemini:gemini-2.5-flash   python examples/01_hello_agent.py
    MODEL=ollama:gemma3:4b          python examples/01_hello_agent.py
    MODEL=hf:Qwen/Qwen3-4B-Instruct-2507  python examples/01_hello_agent.py
"""

from __future__ import annotations

import asyncio

import demo_models
from demo_models import pick
from src import Mind, texts

demo_models.install()

MODEL = pick("MODEL", "thinker")


async def main() -> None:
    mind = Mind(
        "hello",
        MODEL,
        system="You are a careful assistant. Answer in two sentences.",
        run_dir="runs",
    )
    print(mind.describe(), "\n")

    for question in ["What is a digital mind?", "What did I just ask you?"]:
        mind.prompt(question)
        await mind.process()
        print(f"\nyou : {question}")
        print(f"mind: {texts(mind.get_replies())[0]}\n")

    # The transcript is ordinary data. Read it, edit it, save it.
    print(f"transcript holds {len(mind.subject.transcript)} messages")
    print(f"trace written to {mind.run_path}/trace.jsonl")
    print(f"per-module logs in {mind.run_path}/modules/")

    mind.close()


if __name__ == "__main__":
    asyncio.run(main())
