# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclass for the differential-drive action term."""

from __future__ import annotations

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .differential_drive import DifferentialDriveAction


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Config knobs for ``DifferentialDriveAction``.

    Think of this as the "data sheet" taped to the gearbox:

    - which articulation asset name in the scene
    - which joint names are the six driven wheels
    - physical sizes for the skid-steer math
    - how to scale policy outputs into physical ``[v, ω]``
    """

    class_type: type[ActionTerm] = DifferentialDriveAction

    # Scene entity name for Perseverance (must match ArticulationCfg later).
    asset_name: str = "robot"

    # Must match USD / articulation joint names (adjust when your USD is ready).
    wheel_joint_names: list[str] = [
        "wheel_FL_joint",
        "wheel_FR_joint",
        "wheel_ML_joint",
        "wheel_MR_joint",
        "wheel_RL_joint",
        "wheel_RR_joint",
    ]

    # Geometry for kinematics (tune to your asset; ~Perseverance ballpark).
    track_width: float = 2.0  # meters, left-right distance B
    wheel_radius: float = 0.26  # meters (~half of ~0.52 m diameter)

    # Policy often outputs roughly [-1, 1]; we map into physical units:
    #   v = scale[0] * a0 + offset[0]
    #   ω = scale[1] * a1 + offset[1]
    scale: tuple[float, float] = (0.8, 0.8)  # m/s and rad/s peaks from |a|=1
    offset: tuple[float, float] = (0.0, 0.0)

    # Hard safety clips after scaling (physical limits).
    max_linear_vel: float = 0.8  # m/s
    max_angular_vel: float = 0.8  # rad/s
