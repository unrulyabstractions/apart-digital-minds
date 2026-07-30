"""Model access.

    llm.py       LLM           -> BaseLLM, the providers, Cassette
    factory.py   ModelFactory  -> get_llm, taped(...)

`BaseLLM` in `src.dminds.llm` handles timing and thread offload, so a provider
writes one synchronous method. Subclass that unless you need control over the
async call itself, as `Cassette` does.
"""

from .factory import ModelFactory
from .llm import LLM

__all__ = ["LLM", "ModelFactory"]
