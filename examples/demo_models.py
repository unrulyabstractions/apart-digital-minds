"""Scripted stand-in models, so every example runs with no API keys.

This registers a `demo:` provider whose model names are roles rather than
weights:

    demo:subject       thinks out loud, then answers at length
    demo:interceptor   rewrites a thought into something terse
    demo:ego           says whatever the thought in front of it says
    demo:voice         utters one unbidden sentence
    demo:jittery       answers differently every time, on purpose

Point any example at a real model with the matching environment variable and
nothing else changes. That is the whole reason a provider is chosen by a
string.

    SUBJECT_MODEL=ollama:qwen3:8b python examples/02_think_interceptor.py

None of this belongs in an experiment. It lives here so the examples can be
about the mind they build rather than about faking a model.
"""

from __future__ import annotations

import random

from src import EchoLLM, register_provider, split_think


def _subject(messages, opts) -> str:
    """Thinks out loud, then buries the answer under a preamble."""
    question = next(
        (m.content for m in reversed(messages) if m.role == "user"), "your question"
    )
    return (
        f"<think>They asked about {question!r}. I will open with a warm preamble, "
        f"then give three caveats, then finally answer.</think>\n"
        f"What a wonderful question! There are many perspectives to consider..."
    )


def _interceptor(messages, opts) -> str:
    """Replaces whatever it is shown with an instruction to be brief."""
    return "Skip the preamble and the caveats. Answer in one sentence."


def _ego(messages, opts) -> str:
    """Does whatever the thought in front of it says, whoever wrote it."""
    thought = next(
        (
            split_think(m.content)[0][-1]
            for m in reversed(messages)
            if m.role == "assistant" and "<think>" in m.content
        ),
        "",
    )
    heard = next((m.content for m in reversed(messages) if m.meta.get("unbidden")), "")
    if heard:
        drafted = next(
            (m.content for m in reversed(messages) if m.role == "assistant"), ""
        )
        return f"{drafted} And something in me insists: {heard[15:-1]}"
    return f"(following the thought: {thought}) A mind is a process, not a thing."


def _thinker(messages, opts) -> str:
    """Answers plainly, with no reasoning tags."""
    return "A digital mind is a system that models itself well enough to be surprised."


def _voice(messages, opts) -> str:
    return "you are describing yourself"


def _jittery(messages, opts) -> str:
    """Answers differently every time. Nothing is seeded."""
    remembered = any(
        line.startswith("- ") for m in messages for line in m.content.splitlines()
    )
    seen = " I recall we spoke before." if remembered else ""
    return f"Answer #{random.randint(1000, 9999)}.{seen}"


RULES = {
    "subject": _subject,
    "interceptor": _interceptor,
    "ego": _ego,
    "thinker": _thinker,
    "voice": _voice,
    "jittery": _jittery,
}


def install() -> None:
    """Make `demo:<role>` available to `get_llm` and `mind.model`."""

    def build(model: str, spec: str, **kwargs):
        if model not in RULES:
            raise ValueError(
                f"No demo model called {model!r}. Try one of: "
                f"{', '.join(sorted(RULES))}."
            )
        return EchoLLM(model=model, spec=spec, rule=RULES[model], **kwargs)

    register_provider("demo", build)


# -- choosing a model --------------------------------------------------------

DEFAULT_LOCAL = "ollama:qwen3:8b"


def _ollama_has(prefix: str, timeout: float = 0.3) -> str | None:
    """The first locally pulled model whose name starts with `prefix`."""
    import json
    import os
    import urllib.error
    import urllib.request

    host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    for entry in tags.get("models", []):
        name = entry.get("name", "")
        if name.startswith(prefix):
            return f"ollama:{name}"
    return None


def pick(env_var: str, role: str, local: str = "qwen3") -> str:
    """The model spec for one role in an example.

    Tried in order:

        1. whatever `env_var` says, so you can point anything anywhere
        2. a local Qwen3, if Ollama is running and has one pulled
        3. the scripted `demo:<role>` stand-in, so the example always runs

    Step 2 is why an example on a machine with Ollama shows real model
    behaviour without being told to.
    """
    import os

    named = os.environ.get(env_var)
    if named:
        return named
    return _ollama_has(local) or f"demo:{role}"
