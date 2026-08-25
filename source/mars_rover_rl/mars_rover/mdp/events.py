# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Startup / reset helpers that need the live articulation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from mars_rover.assets.robots.perseverance.perseverance import perseverance_usd_path
from mars_rover.mdp.episode_buffers import reset_episode_buffers as reset_mars_episode_buffers
from mars_rover.mdp.kinematics import (
    CHASSIS_BODY_NAME,
    ROVER_ARTICULATION_PRIM,
    ROVER_WHEEL_JOINTS,
    WHEEL_ORDER,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_episode_buffers(env: ManagerBasedRLEnv, env_ids, **kwargs) -> None:
    """EventTerm wrapper so Isaac Lab can bind this function."""
    reset_mars_episode_buffers(env, env_ids, **kwargs)


def validate_wheel_joints(
    env: ManagerBasedRLEnv,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Assert USD, articulation root, chassis body, and six wheel joints."""
    usd = Path(perseverance_usd_path())
    if not usd.is_file():
        raise FileNotFoundError(
            f"Perseverance USD not found at {usd}. "
            "Copy Perseverance_Rover.usd next to perseverance.py or set MARS_ROVER_USD. "
            "The RL asset must be a cleaned robot USD (Rover + materials only), not the "
            "Isaac Sim authoring scene with GroundPlane / env_light / PhysicsScene."
        )

    robot = env.scene[asset_cfg.name]
    all_joints = list(getattr(robot, "joint_names", []))
    body_names = list(getattr(robot, "body_names", []))

    print("[mars_rover] USD:", usd)
    print("[mars_rover] Articulation joints:")
    for name in all_joints:
        print(f"  - {name}")
    print("[mars_rover] Bodies:")
    for name in body_names:
        print(f"  - {name}")
    print(f"[mars_rover] Expected articulation child prim: {ROVER_ARTICULATION_PRIM}")
    print("[mars_rover] Wheel mapping:")
    for token, joint in zip(WHEEL_ORDER, ROVER_WHEEL_JOINTS, strict=True):
        print(f"  {token} → {joint}")
    print(f"[mars_rover] Chassis: {CHASSIS_BODY_NAME}")

    if CHASSIS_BODY_NAME not in body_names:
        raise RuntimeError(
            f"Chassis body '{CHASSIS_BODY_NAME}' not in articulation bodies {body_names}. "
            "Check the USD and CHASSIS_BODY_NAME."
        )

    _ids, names = robot.find_joints(list(ROVER_WHEEL_JOINTS), preserve_order=True)
    resolved = list(names)
    expected = list(ROVER_WHEEL_JOINTS)
    if resolved != expected:
        raise RuntimeError(
            "Wheel joint order mismatch. "
            f"Expected {expected}, resolved {resolved}. "
            "Change ROVER_WHEEL_JOINTS in kinematics.py to match the USD."
        )
    if len(resolved) != 6:
        raise RuntimeError(f"Expected 6 wheel joints {expected}, found {len(resolved)}: {resolved}.")
