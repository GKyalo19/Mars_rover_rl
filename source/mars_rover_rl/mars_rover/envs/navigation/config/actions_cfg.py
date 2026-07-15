# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Action wiring diagram for Perseverance navigation (Phase A).

Connects the policy's 2D output to our DifferentialDriveActionTerm.

Requires Isaac Lab (NVIDIA).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from mars_rover.mdp.actions.actions_cfg import DifferentialDriveActionCfg


@configclass
class ActionsCfg:
    """Action terms available in the env.

    Puzzle piece: the *name* ``chassis_twist`` is how other code refers to this
    term. The policy still outputs a single concatenated action vector; the
    ActionManager slices it according to each term's ``action_dim``.
    """

    chassis_twist = DifferentialDriveActionCfg(
        asset_name="robot",
        # Update these strings to match your articulated USD joint names.
        wheel_joint_names=[
            "wheel_FL_joint",
            "wheel_FR_joint",
            "wheel_ML_joint",
            "wheel_MR_joint",
            "wheel_RL_joint",
            "wheel_RR_joint",
        ],
        track_width=2.0,
        wheel_radius=0.26,
        scale=(0.8, 0.8),
        max_linear_vel=0.8,
        max_angular_vel=0.8,
    )
