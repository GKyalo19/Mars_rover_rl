# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3.0 ProxyArray → torch.Tensor.

Asset/sensor ``.data.*`` properties return ProxyArray. Use ``.torch`` for
PyTorch ops. This helper also accepts a plain tensor so Mac unit tests and
older shims keep working.
"""

from __future__ import annotations

from typing import Any


def as_torch(value: Any):
    """Return a torch tensor view of Lab 3 ProxyArray, or ``value`` unchanged."""
    torch_view = getattr(value, "torch", None)
    if torch_view is None:
        return value
    return torch_view() if callable(torch_view) else torch_view
