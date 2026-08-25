# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclass for the (optional) differential-drive action term.

Baseline v1 uses Isaac Lab's JointVelocityActionCfg, not this term.
This stays as a later 2-D experiment variant.
"""

from __future__ import annotations

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from mars_rover.mdp.kinematics import ROVER_WHEEL_JOINTS

from .differential_drive import DifferentialDriveAction


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Config knobs for ``DifferentialDriveAction`` (not the Baseline v1 default)."""

    class_type: type[ActionTerm] = DifferentialDriveAction

    asset_name: str = "robot"

    wheel_joint_names: list[str] = list(ROVER_WHEEL_JOINTS)

    track_width: float = 2.0
    wheel_radius: float = 0.26

    scale: tuple[float, float] = (0.8, 0.8)
    offset: tuple[float, float] = (0.0, 0.0)

    max_linear_vel: float = 0.8
    max_angular_vel: float = 0.8
