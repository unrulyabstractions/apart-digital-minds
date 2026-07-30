"""The model interface.

    llm.py   LLM   -> BaseLLM, the six providers, Cassette

`BaseLLM` in `src.dminds.llm` handles timing and thread offload, so a provider
writes one synchronous method. Subclass that unless you need control over the
async call itself, as `Cassette` does.
"""

from .llm import LLM

__all__ = ["LLM"]
