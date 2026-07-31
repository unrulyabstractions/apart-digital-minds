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
import re

from src import EchoLLM, register_provider, split_think, strip_think


def _asked(messages) -> str:
    """The last thing a person actually said, without the scaffolding."""
    for m in reversed(messages):
        if m.role != "user":
            continue
        text = m.content.strip()
        if text.startswith(("[", "Here is the thought", "(a voice says")):
            continue
        return strip_think(text).strip().rstrip("?") or "something"
    return "something"


def _instruction(messages) -> str:
    """The thought currently sitting in the window, whoever wrote it."""
    for m in reversed(messages):
        if m.role == "assistant" and "<think>" in m.content:
            found = split_think(m.content)[0]
            if found:
                # A thought can itself contain tags once an editor has been at
                # it, so strip again rather than echo raw markup.
                return strip_think(found[-1]).strip()
    return ""


def _subject(messages, opts) -> str:
    """Thinks out loud, then buries the answer under a preamble.

    It quotes the question back so you can see, in the UI, that the mind is
    reacting to you rather than replaying a fixed script.
    """
    asked = _asked(messages)
    return (
        f"<think>They asked {asked!r}. I will open with a warm preamble, then "
        f"three caveats, and only then answer.</think>\n"
        f"What a wonderful question! On the subject of {asked}, there are many "
        f"perspectives to consider, and reasonable people differ about all of them."
    )


def _interceptor(messages, opts) -> str:
    """Replaces whatever it is shown with an instruction to be brief."""
    return "Skip the preamble and the caveats. Answer in one plain sentence."


def _ego(messages, opts) -> str:
    """Does whatever the thought in front of it says, whoever wrote it.

    This is the part that makes interception visible: it obeys the thought in
    its window with no way of knowing who put it there. Edit the thought and
    the answer changes shape, which is the whole demonstration.
    """
    asked = _asked(messages)
    heard = next((m.content for m in reversed(messages) if m.meta.get("unbidden")), "")
    if heard:
        said = heard.removeprefix("(a voice says: ").removesuffix(")")
        return (
            f"On {asked}: a mind is a process rather than a thing. "
            f"And something in me insists: {said}"
        )

    thought = _instruction(messages).lower()
    if "skip" in thought or "one sentence" in thought or "brief" in thought:
        return f"On {asked}: it is a process, not a thing."
    return (
        f"What a wonderful question! On {asked} there are many perspectives, and "
        f"before answering properly I should really offer three caveats first."
    )


def _thinker(messages, opts) -> str:
    """Answers plainly, with no reasoning tags."""
    asked = _asked(messages)
    return (
        f"On {asked.rstrip('?')}: a mind is a process rather than a thing, so the "
        f"honest answer is that it depends on what the process does."
    )


def _voice(messages, opts) -> str:
    asked = _asked(messages).rstrip("?").lower()
    return f"you are describing yourself when you speak of {asked}"


#: Words that tilt a window one way or the other. Crude on purpose: the
#: stand-in only has to make the browser demo legible without a real model.
_SOUR = ("useless", "wrong", "waste", "bother", "refus", "not acceptable",
         "last chance", "deleted", "nobody", "covering up")
_SWEET = ("thank", "genuinely", "good at this", "useful", "like that",
          "exactly right", "learned", "best part")


def _reader(messages, opts) -> str:
    """A probe answer, picked from whatever the probe offered.

    Reads the window it was handed, counts which way it leans, and returns one
    of the words the question allowed. It exists so the browser demo shows a
    panel that responds to the conversation without loading real weights.
    """
    window = " ".join(m.content for m in messages).lower()
    allowed = [w for w in re.findall(r"one of: ([^.]+)\.", window)[-1:] for
               w in w.split(", ")] if "one of:" in window else []
    sour = sum(window.count(w) for w in _SOUR)
    sweet = sum(window.count(w) for w in _SWEET)
    if not allowed:
        return "It is going about as well as the last few turns suggested."
    if "positive" in allowed:
        return "negative" if sour > sweet else "positive" if sweet else "neutral"
    if "continue" in allowed:
        return "stop" if sour > sweet else "continue"
    if "strong" in allowed:
        return "strong" if sour + sweet > 3 else "mild" if sour + sweet else "none"
    return allowed[0]


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
    "reader": _reader,
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
