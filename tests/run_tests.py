"""Run the test suite with nothing installed.

    python tests/run_tests.py

pytest works too, if you have it. This exists so the suite is runnable on a
bare interpreter, which is the same reason the package has no dependencies.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def modules() -> list[str]:
    """Every `tests/test_*.py`, in a fixed order.

    Discovered rather than listed, because a list is a place to forget a file
    and a forgotten file looks exactly like a passing suite.
    """
    here = Path(__file__).resolve().parent
    return [f"tests.{p.stem}" for p in sorted(here.glob("test_*.py"))]


def main() -> int:
    passed, failed = 0, []

    for module_name in modules():
        module = importlib.import_module(module_name)
        short = module_name.split(".")[-1]
        print(f"\n{short}")
        for name in sorted(vars(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception:
                failed.append((module_name, name, traceback.format_exc()))
                print(f"  FAIL  {name}")
            else:
                passed += 1
                print(f"  ok    {name}")

    print("\n" + "=" * 70)
    if failed:
        for module_name, name, tb in failed:
            print(f"\n--- {module_name}.{name} ---\n{tb}")
        print(f"{passed} passed, {len(failed)} failed")
        return 1
    print(f"{passed} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
