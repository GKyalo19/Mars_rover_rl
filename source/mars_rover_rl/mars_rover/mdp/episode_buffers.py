# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env buffers that must reset with the episode (progress, idle duration)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_buffers(env: ManagerBasedRLEnv) -> None:
    n = env.num_envs
    device = env.device
    if getattr(env, "_mars_prev_goal_distance", None) is None or env._mars_prev_goal_distance.shape[0] != n:
        env._mars_prev_goal_distance = torch.zeros(n, device=device)
        env._mars_progress_initialized = torch.zeros(n, dtype=torch.bool, device=device)
        env._mars_idle_steps = torch.zeros(n, dtype=torch.long, device=device)


def reset_episode_buffers(env: ManagerBasedRLEnv, env_ids, **_kwargs) -> None:
    """Clear progress/idle bookkeeping on episode reset (EventTerm)."""
    _ensure_buffers(env)
    env._mars_progress_initialized[env_ids] = False
    env._mars_idle_steps[env_ids] = 0


def update_progress(env: ManagerBasedRLEnv, current_distance: torch.Tensor) -> torch.Tensor:
    """``previous_distance - current_distance``; zero on the first step after reset."""
    _ensure_buffers(env)
    prev = env._mars_prev_goal_distance
    initialized = env._mars_progress_initialized
    delta = prev - current_distance
    progress = torch.where(initialized, delta, torch.zeros_like(delta))
    env._mars_prev_goal_distance = current_distance.detach()
    env._mars_progress_initialized[:] = True
    return progress


def update_idle_steps(env: ManagerBasedRLEnv, is_idle: torch.Tensor) -> torch.Tensor:
    """Consecutive idle-step counter; resets to 0 when the rover is moving."""
    _ensure_buffers(env)
    idle = is_idle.to(dtype=torch.bool)
    env._mars_idle_steps = torch.where(idle, env._mars_idle_steps + 1, torch.zeros_like(env._mars_idle_steps))
    return env._mars_idle_steps
