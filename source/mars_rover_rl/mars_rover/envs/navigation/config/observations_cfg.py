# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation wiring for Perseverance Baseline v1.

Default policy group is a 19-D proprioceptive vector. Height-scan and vision
groups remain defined for later tracks but are not attached here.
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import mars_rover.envs.navigation.mdp as mdp
from mars_rover.mdp.kinematics import ROVER_WHEEL_JOINTS


@configclass
class ObservationsCfg:
    """Observation groups. PPO reads ``policy`` only."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Concatenated 19-D vector: goal(2)+vxy(2)+wz(1)+wheels(6)+rpy(2)+action(6)."""

        goal_xy_local = ObsTerm(
            func=mdp.observations.goal_xy_local,
            params={"command_name": "target_pose"},
            scale=0.1,
        )
        base_lin_vel_xy = ObsTerm(func=mdp.observations.base_lin_vel_xy)
        base_ang_vel_z = ObsTerm(func=mdp.observations.base_ang_vel_z)
        wheel_velocities = ObsTerm(
            func=mdp.observations.wheel_velocities,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(ROVER_WHEEL_JOINTS),
                    preserve_order=True,
                )
            },
        )
        base_roll_pitch = ObsTerm(
            func=mdp.observations.base_roll_pitch,
            scale=1.0 / math.pi,
        )
        last_action = ObsTerm(func=mdp.observations.last_action)

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class VisionCfg(ObsGroup):
        """Deferred — not attached to ObservationsCfg for Baseline v1."""

        depth_cam = ObsTerm(func=mdp.observations.hazcam_depth, params={"sensor_cfg_name": "hazcam_front"})

        def __post_init__(self) -> None:
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
