"""Interfaces for the three kinds of memory.

    stores.py   MessageStore   -> Transcript
                KeyValueStore  -> Scratchpad
                EpisodicStore  -> Journal
"""

from .stores import EpisodicStore, KeyValueStore, MessageStore

__all__ = ["MessageStore", "KeyValueStore", "EpisodicStore"]
