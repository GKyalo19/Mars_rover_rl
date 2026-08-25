# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP helpers.

Import kinematics freely on any machine::

    from mars_rover.mdp.kinematics import twist_to_wheel_velocities

Isaac-dependent observation/action modules live in sibling files and are
imported only from env configs on the NVIDIA / Isaac Lab environment.
"""

from mars_rover.mdp.kinematics import (
    CHASSIS_BODY_NAME,
    ROVER_WHEEL_JOINTS,
    WHEEL_ACTION_SCALE,
    WHEEL_ORDER,
    WHEEL_VEL_LIMIT,
    twist_to_wheel_velocities,
)

__all__ = [
    "CHASSIS_BODY_NAME",
    "ROVER_WHEEL_JOINTS",
    "WHEEL_ACTION_SCALE",
    "WHEEL_ORDER",
    "WHEEL_VEL_LIMIT",
    "twist_to_wheel_velocities",
]
