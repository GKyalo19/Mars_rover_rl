# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation term functions for Perseverance navigation (Baseline v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from mars_rover.mdp.goal import goal_heading_error, goal_xy_in_base_horizontal_frame
from mars_rover.mdp.kinematics import apply_wheel_direction_signs, roll_pitch_from_quat_xyzw
from mars_rover.mdp.sim_data import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_xy_local(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Goal ``(dx, dy)`` in the rover local horizontal frame, shape ``(N, 2)``."""
    return goal_xy_in_base_horizontal_frame(env, command_name)


def distance_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Euclidean XY distance from rover to goal, shape ``(N, 1)``."""
    xy = goal_xy_in_base_horizontal_frame(env, command_name)
    return torch.norm(xy, p=2, dim=-1, keepdim=True)


def heading_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Signed heading to the goal in the yaw frame, shape ``(N, 1)``."""
    return goal_heading_error(env, command_name).unsqueeze(-1)


def base_lin_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body-frame linear velocity ``(vx, vy)``, shape ``(N, 2)``."""
    return as_torch(env.scene[asset_cfg.name].data.root_lin_vel_b)[:, :2]


def base_ang_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body-frame yaw rate ``wz``, shape ``(N, 1)``."""
    return as_torch(env.scene[asset_cfg.name].data.root_ang_vel_b)[:, 2:3]


def wheel_velocities(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Six wheel joint velocities in canonical order, shape ``(N, 6)``."""
    asset = env.scene[asset_cfg.name]
    vel = as_torch(asset.data.joint_vel)[:, asset_cfg.joint_ids]
    return apply_wheel_direction_signs(vel)


def base_roll_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root roll and pitch, shape ``(N, 2)``."""
    quat = as_torch(env.scene[asset_cfg.name].data.root_quat_w)
    return roll_pitch_from_quat_xyzw(quat)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous processed action, shape ``(N, action_dim)`` (6 for Baseline v1)."""
    return env.action_manager.prev_action


def height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "height_scanner",
    offset: float = 0.5,
) -> torch.Tensor:
    """Flattened local elevation grid. Deferred from Baseline v1 scene."""
    sensor = env.scene.sensors[sensor_cfg_name]
    hit_heights = as_torch(sensor.data.ray_hits_w)[..., 2]
    sensor_height = as_torch(sensor.data.pos_w)[:, 2].unsqueeze(-1)
    return sensor_height - hit_heights - offset


def hazcam_depth(env: ManagerBasedRLEnv, sensor_cfg_name: str = "hazcam_front") -> torch.Tensor:
    """Forward depth image. Deferred from Baseline v1."""
    sensor = env.scene.sensors[sensor_cfg_name]
    depth = sensor.data.output["distance_to_camera"]
    depth = as_torch(depth)
    return torch.clamp(depth, min=0.0, max=10.0) / 10.0
