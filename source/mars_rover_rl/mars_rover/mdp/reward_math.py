# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure reward / termination math (no Isaac Lab). Runnable on a Mac."""

from __future__ import annotations

import torch


def progress_from_distance(
    distance: torch.Tensor,
    *,
    k: float = 0.11,
    max_episode_length: float = 1.0,
) -> torch.Tensor:
    """Legacy distance shaping ``1 / (1 + k * d^2)``. Not used by Baseline v1 rewards."""
    shaped = 1.0 / (1.0 + k * distance * distance)
    return shaped / max_episode_length


def temporal_progress(
    prev_distance: torch.Tensor,
    current_distance: torch.Tensor,
    initialized: torch.Tensor,
) -> torch.Tensor:
    """``prev - current``; zero when the episode has not yet produced a previous distance."""
    delta = prev_distance - current_distance
    return torch.where(initialized, delta, torch.zeros_like(delta))


def idle_from_speed(
    speed: torch.Tensor,
    *,
    threshold: float = 0.05,
    max_episode_length: float = 1.0,
) -> torch.Tensor:
    """Immediate stationary penalty (legacy). Baseline v1 uses consecutive steps."""
    return torch.where(
        speed < threshold,
        torch.ones_like(speed) / max_episode_length,
        torch.zeros_like(speed),
    )


def idle_from_consecutive_steps(
    idle_steps: torch.Tensor,
    *,
    min_steps: int,
    max_episode_length: float = 1.0,
) -> torch.Tensor:
    """Penalty only after ``min_steps`` consecutive idle steps."""
    active = idle_steps >= min_steps
    return torch.where(
        active,
        torch.ones(idle_steps.shape, dtype=torch.float32, device=idle_steps.device) / max_episode_length,
        torch.zeros(idle_steps.shape, dtype=torch.float32, device=idle_steps.device),
    )


def attitude_unsafe_mask(roll_pitch: torch.Tensor, threshold: float) -> torch.Tensor:
    """True when |roll| or |pitch| exceeds ``threshold`` (radians)."""
    return (roll_pitch.abs() > threshold).any(dim=-1)


def actuation_effort_from_wheel_vel(wheel_vel: torch.Tensor) -> torch.Tensor:
    """Mean |wheel ω| — an actuation-effort **proxy**, not electrical energy."""
    return wheel_vel.abs().mean(dim=-1)
