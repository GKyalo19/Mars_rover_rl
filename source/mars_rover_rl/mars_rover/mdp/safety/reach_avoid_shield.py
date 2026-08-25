"""Lightweight reach-avoid shield: a runtime backstop, not a formal HJ solver.

Use at play/eval time (Ch.9) or even during training if early rollouts are
too crash-heavy to learn from. Not required for Ch.7's training loop.
"""

from __future__ import annotations

import torch


def shield_action(
    action: torch.Tensor,
    height_scan: torch.Tensor,
    *,
    clearance_threshold: float = 0.15,
    forward_ray_slice: slice = slice(90, 135),  # rays roughly "ahead" in a 15x15 grid
) -> torch.Tensor:
    """Clip forward velocity when the scan says danger is close ahead.

    Args:
        action: Policy output, shape (N, 2) = [v, ω] (already scaled).
        height_scan: Flattened ray-cast grid, shape (N, 225) (Track A term).
        clearance_threshold: Minimum "safe" relative height/clearance (m).
        forward_ray_slice: Which flattened indices count as "ahead" —
            depends on your grid layout/resolution; verify against Ch.8 §8.2.

    Returns:
        Possibly-modified action, same shape as ``action``.
    """
    forward_clearance = height_scan[:, forward_ray_slice].amin(dim=-1)
    unsafe = forward_clearance < clearance_threshold

    safe_action = action.clone()
    driving_forward = action[:, 0] > 0
    override = unsafe & driving_forward
    safe_action[override, 0] = 0.0  # backup policy: stop forward motion
    return safe_action
