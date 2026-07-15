# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Chassis twist [v, ω] → six wheel angular velocities (skid-steer model).

This module is intentionally free of Isaac Lab imports so you can unit-test
the math on a Mac. The ActionTerm in ``differential_drive.py`` calls into here.
"""

from __future__ import annotations

import torch

# Joint / feature order we standardize on across the project.
# When you name revolute joints in Isaac, keep this order in mind.
WHEEL_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")


def twist_to_wheel_velocities(
    linear_x: torch.Tensor,
    angular_z: torch.Tensor,
    *,
    track_width: float,
    wheel_radius: float,
) -> torch.Tensor:
    """Convert chassis commands to wheel spin rates (rad/s).

    Skid-steer / differential-drive mixing (tank-style):

        v_L = v - ω * (B / 2)
        v_R = v + ω * (B / 2)
        φ̇   = v_wheel / r

    Left wheels (FL, ML, RL) share v_L; right wheels (FR, MR, RR) share v_R.

    Args:
        linear_x: Forward speed v in m/s. Shape ``(N,)`` or ``(N, 1)``.
        angular_z: Yaw rate ω in rad/s. Same shape as ``linear_x``.
        track_width: Distance between left and right contact patches (m), ``B``.
        wheel_radius: Wheel radius ``r`` (m).

    Returns:
        Tensor of shape ``(N, 6)`` — angular velocities for
        FL, FR, ML, MR, RL, RR in rad/s.
    """
    if wheel_radius <= 0.0:
        raise ValueError(f"wheel_radius must be positive, got {wheel_radius}")
    if track_width <= 0.0:
        raise ValueError(f"track_width must be positive, got {track_width}")

    v = linear_x.reshape(-1)
    w = angular_z.reshape(-1)
    if v.shape != w.shape:
        raise ValueError(f"linear_x and angular_z shapes differ: {v.shape} vs {w.shape}")

    half_b = track_width * 0.5
    v_left = v - w * half_b
    v_right = v + w * half_b

    w_left = v_left / wheel_radius
    w_right = v_right / wheel_radius

    # Stack into (N, 6) matching WHEEL_ORDER.
    return torch.stack([w_left, w_right, w_left, w_right, w_left, w_right], dim=-1)
