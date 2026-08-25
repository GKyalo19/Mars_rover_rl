"""Reward terms for Perseverance navigation (Baseline v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from mars_rover.mdp.contact import max_net_contact_force
from mars_rover.mdp.episode_buffers import update_idle_steps, update_progress
from mars_rover.mdp.goal import goal_distance_xy
from mars_rover.mdp.kinematics import (
    CHASSIS_CONTACT_THRESHOLD,
    IDLE_STEPS_THRESHOLD,
    UNSAFE_ATTITUDE_THRESHOLD,
    apply_wheel_direction_signs,
    roll_pitch_from_quat_xyzw,
)
from mars_rover.mdp.reward_math import actuation_effort_from_wheel_vel, attitude_unsafe_mask
from mars_rover.mdp.sim_data import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """True temporal progress: previous distance minus current distance (meters)."""
    current = goal_distance_xy(env, command_name)
    return update_progress(env, current)


def reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
    distance_threshold: float = 0.25,
) -> torch.Tensor:
    """Sparse success bonus when within the same distance used by termination."""
    distance = goal_distance_xy(env, command_name)
    time_left_frac = (env.max_episode_length - env.episode_length_buf) / env.max_episode_length
    success = distance < distance_threshold
    return torch.where(success, 2.0 * time_left_frac, torch.zeros_like(distance))


def oscillation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize jerky changes between consecutive actions (deferred from Baseline v1 cfg)."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return delta.square().mean(dim=-1) / env.max_episode_length


def reverse_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty when the mean wheel command is negative (deferred from Baseline v1 cfg)."""
    mean_cmd = env.action_manager.action.mean(dim=-1)
    return torch.where(
        mean_cmd < 0.0,
        torch.ones_like(mean_cmd) / env.max_episode_length,
        torch.zeros_like(mean_cmd),
    )


def rotation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on |yaw rate| (deferred from Baseline v1 cfg)."""
    wz = as_torch(env.scene[asset_cfg.name].data.root_ang_vel_b)[:, 2]
    return wz.abs() / env.max_episode_length


def actuation_effort_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mean |wheel ω| proxy — not electrical energy consumption."""
    asset = env.scene[asset_cfg.name]
    vel = as_torch(asset.data.joint_vel)[:, asset_cfg.joint_ids]
    effort = actuation_effort_from_wheel_vel(apply_wheel_direction_signs(vel))
    return effort / env.max_episode_length


def safety_attitude(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = UNSAFE_ATTITUDE_THRESHOLD,
) -> torch.Tensor:
    """1 when |roll| or |pitch| exceeds ``threshold`` rad (~0.7 ≈ 40°)."""
    quat = as_torch(env.scene[asset_cfg.name].data.root_quat_w)
    rp = roll_pitch_from_quat_xyzw(quat)
    unsafe = attitude_unsafe_mask(rp, threshold)
    return unsafe.to(dtype=torch.float32)


def idle_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_threshold: float = 0.05,
    idle_steps_threshold: int = IDLE_STEPS_THRESHOLD,
) -> torch.Tensor:
    """Penalty after ``idle_steps_threshold`` consecutive near-zero planar-speed steps."""
    vel_xy = as_torch(env.scene[asset_cfg.name].data.root_lin_vel_w)[:, :2]
    speed = torch.norm(vel_xy, p=2, dim=-1)
    steps = update_idle_steps(env, speed < speed_threshold)
    active = steps >= idle_steps_threshold
    return torch.where(
        active,
        torch.ones(env.num_envs, device=env.device) / env.max_episode_length,
        torch.zeros(env.num_envs, device=env.device),
    )


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = CHASSIS_CONTACT_THRESHOLD,
) -> torch.Tensor:
    """Penalty when chassis net contact force exceeds the tunable threshold (N)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    max_force = max_net_contact_force(contact_sensor)
    in_collision = max_force > threshold
    return torch.where(
        in_collision,
        torch.ones(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant per-step cost to discourage endless wandering."""
    return torch.ones(env.num_envs, device=env.device) / env.max_episode_length
