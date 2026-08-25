# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Single authoritative goal source for observations, rewards, and terminations.

The command manager owns per-env goal pose. Every MDP term that needs the
goal must go through these helpers so obs and reward cannot diverge.

World goal is reconstructed from ``pose_command_b`` plus the live root pose so
the first observation after reset is not a stale ``pose_command_w`` cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mars_rover.mdp.kinematics import xy_world_to_base_horizontal, yaw_from_quat_xyzw
from mars_rover.mdp.sim_data import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_xy_world(env: ManagerBasedRLEnv, command_name: str = "target_pose") -> torch.Tensor:
    """Goal XY in world, shape ``(N, 2)``."""
    robot = env.scene["robot"]
    root_pos = as_torch(robot.data.root_pos_w)
    root_quat = as_torch(robot.data.root_quat_w)
    term = env.command_manager.get_term(command_name)

    pos_b = None
    if hasattr(term, "pose_command_b"):
        pose_b = as_torch(term.pose_command_b)
        pos_b = pose_b[:, :3]
    else:
        command = env.command_manager.get_command(command_name)
        if command.shape[-1] >= 3:
            pos_b = command[:, :3]

    if pos_b is not None:
        from isaaclab.utils.math import quat_apply

        offset_w = quat_apply(root_quat, pos_b)
        return root_pos[:, :2] + offset_w[:, :2]

    command = env.command_manager.get_command(command_name)
    return root_pos[:, :2] + command[:, :2]


def goal_xy_in_base_horizontal_frame(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Goal ``(dx, dy)`` in the rover local horizontal frame, shape ``(N, 2)``."""
    robot = env.scene["robot"]
    goal_xy_w = goal_xy_world(env, command_name)
    robot_xy = as_torch(robot.data.root_pos_w)[:, :2]
    yaw = yaw_from_quat_xyzw(as_torch(robot.data.root_quat_w))
    return xy_world_to_base_horizontal(goal_xy_w, robot_xy, yaw)


def goal_distance_xy(env: ManagerBasedRLEnv, command_name: str = "target_pose") -> torch.Tensor:
    """Planar distance to the goal, shape ``(N,)``."""
    return torch.norm(goal_xy_in_base_horizontal_frame(env, command_name), p=2, dim=-1)


def goal_heading_error(env: ManagerBasedRLEnv, command_name: str = "target_pose") -> torch.Tensor:
    """Signed heading to the goal in the yaw frame, shape ``(N,)``."""
    xy = goal_xy_in_base_horizontal_frame(env, command_name)
    return torch.atan2(xy[:, 1], xy[:, 0])
