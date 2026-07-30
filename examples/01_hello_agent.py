"""One agent, one prompt, and a full trace on disk.

Run it:
    python examples/01_hello_agent.py

Swap the model without touching anything else:
    MODEL=openai:gpt-5              python examples/01_hello_agent.py
    MODEL=anthropic:claude-opus-5   python examples/01_hello_agent.py
    MODEL=gemini:gemini-2.5-flash   python examples/01_hello_agent.py
    MODEL=ollama:qwen3:8b           python examples/01_hello_agent.py
    MODEL=ollama:gemma3:4b          python examples/01_hello_agent.py
    MODEL=hf:Qwen/Qwen3-4B-Instruct-2507  python examples/01_hello_agent.py
"""

from __future__ import annotations

import asyncio
import os

from src.api import Agent, Mind, get_llm, texts

MODEL = os.environ.get("MODEL", "echo:")


async def main() -> None:
    mind = Mind("hello", run_dir="runs")

    mind.add(
        Agent(
            "assistant",
            get_llm(MODEL),
            system="You are a careful assistant. Answer in two sentences.",
        )
    )

    print(mind.describe(), "\n")

    for question in ["What is a digital mind?", "What did I just ask you?"]:
        replies = await mind.prompt(question)
        print(f"\nyou : {question}")
        print(f"mind: {texts(replies)[0]}\n")

    # The transcript is ordinary data. Read it, edit it, save it.
    agent = mind.modules["assistant"]
    print(f"transcript holds {len(agent.transcript)} messages")
    print(f"trace written to {mind.run_path}/trace.jsonl")
    print(f"per-module logs in {mind.run_path}/modules/")

    mind.close()


if __name__ == "__main__":
    asyncio.run(main())
