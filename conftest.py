"""Pytest configuration.

The presence of this file at the project root is what puts the root directory
on `sys.path`, so that `tests/` can import `config`, `data` and `models` the
same way `app.py` and `python -m evaluation.evaluate` do.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
