"""One set of options has to work on every provider.

A template-only option means something to a provider that renders a chat
template locally and is a hard error to a hosted API. Swapping a local model
for a hosted one is meant to be a one-string change, so the option has to
survive the swap.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.types import GenOptions  # noqa: E402
from src.dminds.llm.base import sampling_extra  # noqa: E402


def test_template_options_are_dropped_for_hosted_providers():
    opts = GenOptions(extra={"chat_template_kwargs": {"enable_thinking": False},
                             "top_p": 0.9})
    kept = sampling_extra(opts)
    assert "chat_template_kwargs" not in kept
    assert kept["top_p"] == 0.9


def test_nothing_else_is_dropped():
    opts = GenOptions(extra={"top_p": 0.9, "frequency_penalty": 0.1})
    assert sampling_extra(opts) == {"top_p": 0.9, "frequency_penalty": 0.1}


def test_every_hosted_provider_uses_the_filter():
    root = Path(__file__).resolve().parent.parent / "src/dminds/llm/providers"
    for name in ("openai_.py", "anthropic_.py", "gemini_.py", "ollama_.py"):
        source = (root / name).read_text()
        assert "opts.extra" not in source, f"{name} still forwards opts.extra raw"
        assert "sampling_extra" in source, f"{name} does not filter its options"
