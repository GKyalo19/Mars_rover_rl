# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Perseverance ArticulationCfg for Isaac Lab.

Use a **cleaned RL USD** named ``Perseverance_Rover.usd``:

- Keep ``Rover`` + ``_materials``
- Remove ``GroundPlane``, ``env_light``, and ``PhysicsScene`` from the
  Isaac Sim authoring scene before feeding it to ``UsdFileCfg``.

Place that file next to this module, or set ``MARS_ROVER_USD``.

The authoring scene (``Mars Rover Suspension Fixed.usd``) is the master, not
the Isaac Lab spawn asset — do not point ``UsdFileCfg`` at it.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from mars_rover.mdp.kinematics import (
    CHASSIS_BODY_NAME,
    ROVER_ARTICULATION_PRIM,
    ROVER_WHEEL_JOINTS,
    WHEEL_VEL_LIMIT,
)

_USD_NAME = "Perseverance_Rover.usd"
_PACKAGE_DIR = Path(__file__).resolve().parent


def perseverance_usd_path() -> str:
    """Resolve the rover USD without hard-coding a machine-specific path.

    Does not raise if the file is missing so Mac-side imports of this module
    (inside Isaac configs) stay possible. Spawning will fail later with a
    clear USD error if the file is absent.
    """
    env_path = os.environ.get("MARS_ROVER_USD")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str((_PACKAGE_DIR / _USD_NAME).resolve())


PERSEVERANCE_CFG = ArticulationCfg(
    # Spawn prim is {ENV}/Robot; the articulated rover lives one level down.
    articulation_root_prim_path=f"/{ROVER_ARTICULATION_PRIM}",
    spawn=sim_utils.UsdFileCfg(
        usd_path=perseverance_usd_path(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        rot=(0.0, 0.0, 0.0, 1.0),  # Isaac Lab 3.0 identity: (x, y, z, w)
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=list(ROVER_WHEEL_JOINTS),
            # PhysX drive limits (not a Python motor model).
            effort_limit_sim=80.0,
            velocity_limit_sim=WHEEL_VEL_LIMIT,
            stiffness=0.0,  # velocity drive: P-gain off
            damping=50.0,  # D-gain tracks the velocity target
        ),
    },
)

# Re-export so scene/configs can import chassis name from the asset module.
CHASSIS_PRIM = CHASSIS_BODY_NAME
