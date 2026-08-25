# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Action wiring for Perseverance Baseline v1.

Default: Isaac Lab JointVelocityActionCfg → six wheel velocity targets.
The 2-D differential-drive term remains in the package as a later experiment.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg
from isaaclab.utils import configclass

from mars_rover.mdp.kinematics import ROVER_WHEEL_JOINTS, WHEEL_VEL_LIMIT, wheel_action_scale_map


@configclass
class ActionsCfg:
    """Policy output is 6-D: one angular-velocity command per wheel."""

    wheel_vel = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(ROVER_WHEEL_JOINTS),
        scale=wheel_action_scale_map(),
        preserve_order=True,
        use_default_offset=False,
        clip={".*": (-WHEEL_VEL_LIMIT, WHEEL_VEL_LIMIT)},
    )
