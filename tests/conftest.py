# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Make ``mars_rover`` importable without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "source" / "mars_rover_rl"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
