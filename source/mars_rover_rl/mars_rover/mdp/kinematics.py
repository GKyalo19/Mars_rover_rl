# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Chassis twist [v, ω] → six wheel angular velocities (skid-steer model).

This module is intentionally free of Isaac Lab imports so you can unit-test
the math on a Mac. The optional DifferentialDrive ActionTerm calls into here.

Canonical wheel order (spec Baseline v1): left side then right side.
"""

from __future__ import annotations

import torch

# Policy action[i] maps to ROVER_WHEEL_JOINTS[i] when preserve_order=True.
# Names are from the authored Perseverance USD (not conceptual chapter names).
WHEEL_ORDER = ("FL", "ML", "RL", "FR", "MR", "RR")
ROVER_WHEEL_JOINTS = (
    "Joint_Wheel_FL",
    "Joint_Wheel_ML",
    "Joint_Wheel_RL",
    "Joint_Wheel_FR",
    "Joint_Wheel_MR",
    "Joint_Wheel_RR",
)
CHASSIS_BODY_NAME = "Body"
ROVER_ARTICULATION_PRIM = "Rover"

# Default +1 until Gate B measures joint axis signs. Do not guess opposites.
WHEEL_DIRECTION_SIGNS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

# JointVelocityActionCfg scale / clip (rad/s). 3.08 ≈ 0.8 m/s at r=0.26 m.
WHEEL_ACTION_SCALE = 3.0
WHEEL_VEL_LIMIT = 3.08

# Tunable "rover is probably in trouble" attitude limit (~40°). Not a physical constant.
UNSAFE_ATTITUDE_THRESHOLD = 0.7
# Initial chassis-contact threshold (N). Measure on ASUS, then document the chosen value.
CHASSIS_CONTACT_THRESHOLD = 50.0
# Policy steps (~1/15 s) of near-zero speed before idle penalty.
IDLE_STEPS_THRESHOLD = 30


def scale_and_clip_wheel_actions(
    raw: torch.Tensor,
    *,
    scale: float = WHEEL_ACTION_SCALE,
    clip: float = WHEEL_VEL_LIMIT,
) -> torch.Tensor:
    """Map raw policy outputs to wheel velocity targets (rad/s).

    Isaac Lab's JointVelocityActionCfg performs this natively; this helper
    documents the Baseline v1 contract for Mac unit tests.
    """
    return torch.clamp(raw * scale, -clip, clip)


def chassis_prim_path(env_ns: str = "{ENV_REGEX_NS}") -> str:
    """Prim path of the chassis rigid body under the spawned Robot asset."""
    return f"{env_ns}/Robot/{ROVER_ARTICULATION_PRIM}/{CHASSIS_BODY_NAME}"


def wheel_action_scale_map() -> dict[str, float]:
    """Per-joint velocity scale, including measured direction signs."""
    return {
        name: WHEEL_ACTION_SCALE * sign
        for name, sign in zip(ROVER_WHEEL_JOINTS, WHEEL_DIRECTION_SIGNS, strict=True)
    }


def apply_wheel_direction_signs(wheel_vel: torch.Tensor) -> torch.Tensor:
    """Flip observed wheel ω so + means the same physical roll as +action."""
    signs = torch.tensor(WHEEL_DIRECTION_SIGNS, dtype=wheel_vel.dtype, device=wheel_vel.device)
    return wheel_vel * signs


def action_index_to_joint() -> tuple[tuple[int, str, str], ...]:
    """``action[i] → (token, USD joint name)`` with preserve_order=True."""
    return tuple(
        (i, token, joint)
        for i, (token, joint) in enumerate(zip(WHEEL_ORDER, ROVER_WHEEL_JOINTS, strict=True))
    )


def yaw_from_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rotation about world Z) from Isaac Lab 3.0 ``(x, y, z, w)`` quaternions.

    Args:
        quat: Shape ``(N, 4)`` in ``(x, y, z, w)`` order (Lab 3.0 / PhysX / Warp).

    Returns:
        Yaw in radians, shape ``(N,)``.
    """
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_from_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Extract roll and pitch from Isaac Lab 3.0 ``(x, y, z, w)`` quaternions.

    Args:
        quat: Shape ``(N, 4)`` in ``(x, y, z, w)`` order.

    Returns:
        Tensor of shape ``(N, 2)`` — ``[roll, pitch]`` in radians.
    """
    x, y, z, w = quat.unbind(-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(sinp.clamp(-1.0, 1.0))
    return torch.stack([roll, pitch], dim=-1)


def xy_world_to_base_horizontal(
    goal_xy_w: torch.Tensor,
    robot_xy_w: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Goal offset in the rover's local horizontal frame (forward, left).

    Uses yaw only so roll/pitch do not mix into ``dx, dy``.

    Args:
        goal_xy_w: Goal XY in world, shape ``(N, 2)``.
        robot_xy_w: Rover XY in world, shape ``(N, 2)``.
        yaw: Rover heading about world Z, shape ``(N,)``.

    Returns:
        ``(N, 2)`` with ``dx`` forward and ``dy`` left in the yaw frame.
    """
    delta = goal_xy_w - robot_xy_w
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    dx = cos_y * delta[:, 0] + sin_y * delta[:, 1]
    dy = -sin_y * delta[:, 0] + cos_y * delta[:, 1]
    return torch.stack([dx, dy], dim=-1)


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
        FL, ML, RL, FR, MR, RR in rad/s.
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

    # Stack into (N, 6) matching WHEEL_ORDER: FL, ML, RL, FR, MR, RR.
    return torch.stack([w_left, w_left, w_left, w_right, w_right, w_right], dim=-1)
