# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation term functions for Perseverance navigation (Phase A).

Isaac Lab's ObservationManager calls each function every step:

    term(env, **params) -> Tensor[num_envs, dim]

These imports require Isaac Lab (NVIDIA machine). On a Mac, read the logic;
runtime execution waits until the full env exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Euclidean XY distance from rover to goal, shape (N, 1).

    Puzzle piece: this is the main "how close am I?" signal. Chapter 5 will
    reward *decreasing* this value (progress).

    Implementation note:
        When CommandsCfg is wired in Chapter 6, ``command_manager.get_command``
        typically returns goal pose in the robot / env frame. Until then this
        function shows the intended API and returns zeros so shapes stay valid
        in dry configs.
    """
    # --- Chapter 6 will replace this stub with real command reads, e.g.:
    # command = env.command_manager.get_command(command_name)
    # goal_xy = command[:, :2]
    # distance = torch.norm(goal_xy, p=2, dim=-1, keepdim=True)
    # return distance
    return torch.zeros(env.num_envs, 1, device=env.device)


def heading_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Signed angle toward the goal in the body frame, shape (N, 1), radians.

    Puzzle piece: even with small distance, a huge heading error means you are
    pointing the wrong way. Policies learn to turn before charging forward.

    Ideal implementation (Ch.6+):

        goal_xy = command[:, :2]
        heading = torch.atan2(goal_xy[:, 1], goal_xy[:, 0]).unsqueeze(-1)
    """
    return torch.zeros(env.num_envs, 1, device=env.device)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous action ``[v, ω]``, shape (N, 2).

    Puzzle piece: helps the Markov property — the net "remembers" what it
    commanded last step, which supports smooth driving (and Ch.5 oscillation
    penalties compare action deltas).
    """
    return env.action_manager.action
