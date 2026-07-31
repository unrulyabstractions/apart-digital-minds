"""Minds you can run. One definition per file.

Every module here exposes the same three things, so a script or the UI can
list what is available and build any of it without knowing the details:

    TITLE    one line, for a menu
    ABOUT    the shape of the pipeline
    ROLES    the extra models it takes: kwarg -> the role that model plays
    build(model, **kwargs) -> Mind

    from minds import available, load
    mind = load("interceptor").build("openai:gpt-5")
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent

#: Modules here that hold shared pieces rather than a mind of their own.
_NOT_MINDS = {"parts"}


def names() -> list[str]:
    """Every mind that can be built, alphabetically."""
    return sorted(
        m.name
        for m in pkgutil.iter_modules([str(HERE)])
        if not m.name.startswith("_") and m.name not in _NOT_MINDS
    )


def load(name: str) -> ModuleType:
    """One mind module, by name."""
    if name not in names():
        raise ValueError(f"No mind called {name!r}. Available: {', '.join(names())}.")
    return importlib.import_module(f"{__name__}.{name}")


def available() -> list[dict]:
    """Every mind with its title and shape, for a menu."""
    out = []
    for name in names():
        module = load(name)
        out.append(
            {
                "name": name,
                "title": getattr(module, "TITLE", name),
                "about": getattr(module, "ABOUT", ""),
                "roles": dict(getattr(module, "ROLES", {})),
            }
        )
    return out
