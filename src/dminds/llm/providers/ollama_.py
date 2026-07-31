"""Ollama: the easiest way to run Qwen, Gemma, or Llama locally.

This talks to the Ollama HTTP server with the standard library, so it adds no
dependency. Start the server and pull a model first:

    ollama pull qwen3:8b
    ollama pull gemma3:4b
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ..base import BaseLLM, sampling_extra
from ....api.types import ChatMessage, Completion, GenOptions, Usage

DEFAULT_HOST = "http://localhost:11434"


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "qwen3:8b",
        spec: str | None = None,
        host: str | None = None,
        timeout: float = 300.0,
    ):
        super().__init__(model, spec)
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout

    def _chat(self, messages: list[ChatMessage], opts: GenOptions) -> Completion:
        options: dict[str, Any] = {
            "temperature": opts.temperature,
            "num_predict": opts.max_tokens,
        }
        if opts.stop:
            options["stop"] = opts.stop
        if opts.seed is not None:
            options["seed"] = opts.seed
        options.update(sampling_extra(opts))

        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": m.role, "content": m.content} for m in messages
                ],
                "stream": False,
                "options": options,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Start it with `ollama serve`. Original error: {exc}"
            ) from exc

        return Completion(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            usage=Usage(
                input_tokens=data.get("prompt_eval_count", 0) or 0,
                output_tokens=data.get("eval_count", 0) or 0,
            ),
            finish_reason=data.get("done_reason", ""),
            raw=data,
        )
