"""Make the repo importable when running under pytest from anywhere."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
