"""Anthropic Claude."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseLLM, merge_consecutive, split_system
from ....api.types import ChatMessage, Completion, GenOptions, Usage


class AnthropicLLM(BaseLLM):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        spec: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        **client_kwargs: Any,
    ):
        super().__init__(model, spec)
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.client_kwargs = client_kwargs
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ImportError(
                    "The anthropic package is required. Install it with: "
                    "pip install anthropic"
                ) from exc

            key = self.api_key or os.environ.get(self.api_key_env)
            if not key:
                raise ValueError(
                    f"{self.api_key_env} is not set. Export it, or pass api_key=..."
                )
            self._client = Anthropic(api_key=key, **self.client_kwargs)
        return self._client

    def _chat(self, messages: list[ChatMessage], opts: GenOptions) -> Completion:
        client = self._get_client()
        system_prompt, rest = split_system(messages)
        # The Messages API rejects two same-role turns in a row, which context
        # editing produces routinely.
        rest = merge_consecutive(rest)
        if not rest:
            rest = [ChatMessage("user", "(continue)")]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": opts.max_tokens,
            "temperature": opts.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in rest],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if opts.stop:
            kwargs["stop_sequences"] = opts.stop
        kwargs.update(opts.extra)

        response = client.messages.create(**kwargs)
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return Completion(
            text=text,
            model=getattr(response, "model", self.model),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason or "",
            raw=response,
        )
