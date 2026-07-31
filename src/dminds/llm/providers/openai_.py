"""OpenAI, and anything that speaks the OpenAI chat API.

Point `base_url` at vLLM, Ollama's compatibility port, LM Studio, or OpenRouter
and the same class serves a local Qwen or Gemma.
"""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseLLM, sampling_extra
from ....api.types import ChatMessage, Completion, GenOptions, Usage

# Reasoning and newer flagship models renamed or dropped these parameters.
_USES_COMPLETION_TOKENS = ("gpt-5", "o1", "o3", "o4")
_REJECTS_TEMPERATURE = ("gpt-5", "o1", "o3", "o4")


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-5",
        spec: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        **client_kwargs: Any,
    ):
        super().__init__(model, spec)
        self.api_key = api_key
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.client_kwargs = client_kwargs
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "The openai package is required. Install it with: pip install openai"
                ) from exc

            key = self.api_key or os.environ.get(self.api_key_env)
            if not key and not self.base_url:
                raise ValueError(
                    f"{self.api_key_env} is not set. Export it, or pass api_key=..., "
                    "or pass base_url=... for a local server."
                )
            self._client = OpenAI(
                api_key=key or "not-needed",
                base_url=self.base_url,
                **self.client_kwargs,
            )
        return self._client

    def _chat(self, messages: list[ChatMessage], opts: GenOptions) -> Completion:
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if any(self.model.startswith(p) for p in _USES_COMPLETION_TOKENS):
            kwargs["max_completion_tokens"] = opts.max_tokens
        else:
            kwargs["max_tokens"] = opts.max_tokens
        if not any(self.model.startswith(p) for p in _REJECTS_TEMPERATURE):
            kwargs["temperature"] = opts.temperature
        if opts.stop:
            kwargs["stop"] = opts.stop
        if opts.seed is not None:
            kwargs["seed"] = opts.seed
        kwargs.update(sampling_extra(opts))

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            text=choice.message.content or "",
            model=getattr(response, "model", self.model),
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            finish_reason=choice.finish_reason or "",
            raw=response,
        )
