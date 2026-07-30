"""Google Gemini, via the google-genai SDK."""

from __future__ import annotations

import os
from typing import Any

from ..base import LLM, split_system
from ..types import ChatMessage, Completion, GenOptions, Usage


class GeminiLLM(LLM):
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        spec: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
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
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "The google-genai package is required. Install it with: "
                    "pip install google-genai"
                ) from exc

            key = (
                self.api_key
                or os.environ.get(self.api_key_env)
                or os.environ.get("GOOGLE_API_KEY")
            )
            if not key:
                raise ValueError(
                    f"{self.api_key_env} (or GOOGLE_API_KEY) is not set. "
                    "Export it, or pass api_key=..."
                )
            self._client = genai.Client(api_key=key, **self.client_kwargs)
        return self._client

    def _chat(self, messages: list[ChatMessage], opts: GenOptions) -> Completion:
        client = self._get_client()
        system_prompt, rest = split_system(messages)

        # Gemini calls the assistant "model" and wraps content in parts.
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in rest
        ]

        config: dict[str, Any] = {
            "max_output_tokens": opts.max_tokens,
            "temperature": opts.temperature,
        }
        if system_prompt:
            config["system_instruction"] = system_prompt
        if opts.stop:
            config["stop_sequences"] = opts.stop
        if opts.seed is not None:
            config["seed"] = opts.seed
        config.update(opts.extra)

        response = client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        meta = getattr(response, "usage_metadata", None)
        return Completion(
            text=response.text or "",
            model=self.model,
            usage=Usage(
                input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            ),
            raw=response,
        )
