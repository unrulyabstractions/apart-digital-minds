"""One function to get any model: `get_llm("provider:model")`.

Swapping backends is a string change and nothing else.

    get_llm("echo:")                              # no keys, deterministic
    get_llm("openai:gpt-5")
    get_llm("anthropic:claude-opus-5")
    get_llm("gemini:gemini-2.5-flash")
    get_llm("ollama:qwen3:8b")                    # local Qwen
    get_llm("ollama:gemma3:4b")                   # local Gemma
    get_llm("hf:Qwen/Qwen3-4B-Instruct-2507")     # local, weights in-process
    get_llm("vllm:Qwen/Qwen3-8B", base_url=...)   # any OpenAI-compatible server

Short aliases exist for quick work: "echo", "qwen", "gemma", "claude", "gpt",
"gemini". Register your own provider with `register_provider`.
"""

from __future__ import annotations

from typing import Any, Callable

from ...api.models import LLM

#: provider name -> factory(model, spec, **kwargs) -> LLM
_PROVIDERS: dict[str, Callable[..., LLM]] = {}

#: bare word -> full spec
ALIASES = {
    "echo": "echo:echo",
    "qwen": "ollama:qwen3:8b",
    "gemma": "ollama:gemma3:4b",
    "claude": "anthropic:claude-sonnet-5",
    "gpt": "openai:gpt-5",
    "gemini": "gemini:gemini-2.5-flash",
}

#: used when a spec names a provider but no model, e.g. "openai:"
DEFAULT_MODELS = {
    "echo": "echo",
    "openai": "gpt-5",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen3:8b",
    "hf": "Qwen/Qwen3-4B-Instruct-2507",
    "vllm": "Qwen/Qwen3-8B",
}


def register_provider(name: str, factory: Callable[..., LLM]) -> None:
    """Add a backend. The factory is called as factory(model, spec, **kwargs)."""
    _PROVIDERS[name] = factory


def available_providers() -> list[str]:
    _install_builtins()
    return sorted(_PROVIDERS)


def parse_spec(spec: str) -> tuple[str, str]:
    """Split "provider:model" into its parts.

    Only the first colon separates, so "ollama:qwen3:8b" keeps its tag and
    "hf:Qwen/Qwen3-4B" keeps its slash path.
    """
    spec = ALIASES.get(spec.strip(), spec.strip())
    if ":" not in spec:
        raise ValueError(
            f"Model spec {spec!r} needs a provider prefix, for example "
            f"'openai:gpt-5' or 'ollama:qwen3:8b'. Known aliases: "
            f"{', '.join(sorted(ALIASES))}."
        )
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    return provider, model.strip() or DEFAULT_MODELS.get(provider, "")


def get_llm(spec: str, **kwargs: Any) -> LLM:
    """Build a model from a spec string. Extra kwargs go to the provider."""
    _install_builtins()
    provider, model = parse_spec(spec)
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Available: "
            f"{', '.join(available_providers())}. "
            f"Add your own with dmind.llm.register_provider."
        )
    if not model:
        raise ValueError(f"Spec {spec!r} names no model and has no default.")
    return _PROVIDERS[provider](model=model, spec=f"{provider}:{model}", **kwargs)


_installed = False


def _install_builtins() -> None:
    """Register the shipped providers.

    Each factory imports its module lazily, so a missing SDK only matters when
    you actually ask for that provider.
    """
    global _installed
    if _installed:
        return
    _installed = True

    def echo(**kw):
        from .providers.echo import EchoLLM

        return EchoLLM(**kw)

    def openai(**kw):
        from .providers.openai_ import OpenAILLM

        return OpenAILLM(**kw)

    def anthropic(**kw):
        from .providers.anthropic_ import AnthropicLLM

        return AnthropicLLM(**kw)

    def gemini(**kw):
        from .providers.gemini_ import GeminiLLM

        return GeminiLLM(**kw)

    def ollama(**kw):
        from .providers.ollama_ import OllamaLLM

        return OllamaLLM(**kw)

    def hf(**kw):
        from .providers.hf_ import HFLLM

        return HFLLM(**kw)

    def vllm(**kw):
        from .providers.openai_ import OpenAILLM

        kw.setdefault("base_url", "http://localhost:8000/v1")
        kw.setdefault("api_key_env", "VLLM_API_KEY")
        return OpenAILLM(**kw)

    for name, factory in [
        ("echo", echo),
        ("openai", openai),
        ("anthropic", anthropic),
        ("gemini", gemini),
        ("ollama", ollama),
        ("hf", hf),
        ("vllm", vllm),
    ]:
        _PROVIDERS.setdefault(name, factory)
