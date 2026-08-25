"""Termination terms for Perseverance navigation (Baseline v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from mars_rover.mdp.goal import goal_distance_xy
from mars_rover.mdp.kinematics import (
    CHASSIS_CONTACT_THRESHOLD,
    UNSAFE_ATTITUDE_THRESHOLD,
    roll_pitch_from_quat_xyzw,
)
from mars_rover.mdp.reward_math import attitude_unsafe_mask
from mars_rover.mdp.contact import max_net_contact_force
from mars_rover.mdp.sim_data import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End when the episode length budget is consumed (truncation)."""
    return env.episode_length_buf >= env.max_episode_length


def goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
    distance_threshold: float = 0.25,
) -> torch.Tensor:
    """End successfully when within distance_threshold of the goal XY."""
    return goal_distance_xy(env, command_name) < distance_threshold


def illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = CHASSIS_CONTACT_THRESHOLD,
) -> torch.Tensor:
    """End when chassis net contact force exceeds threshold (not wheel-ground)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return max_net_contact_force(contact_sensor) > threshold


def farther_than_allowed(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
    max_distance: float = 25.0,
) -> torch.Tensor:
    """End if the rover is far from the (shared) goal — likely stuck/lost."""
    return goal_distance_xy(env, command_name) > max_distance


def unsafe_attitude(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = UNSAFE_ATTITUDE_THRESHOLD,
) -> torch.Tensor:
    """End when |roll| or |pitch| exceeds ``threshold`` rad (default 0.7)."""
    quat = as_torch(env.scene[asset_cfg.name].data.root_quat_w)
    rp = roll_pitch_from_quat_xyzw(quat)
    return attitude_unsafe_mask(rp, threshold)
