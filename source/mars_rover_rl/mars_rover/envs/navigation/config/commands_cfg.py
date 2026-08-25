"""Goal commands for Perseverance navigation.

Phase A uses Isaac Lab's terrain-aware position command utilities when available.
If API names shift slightly in your Lab 3.x build, compare with:

  isaaclab.envs.mdp.commands

and with RLRoverLab's TerrainBasedPositionCommandCfg idea:
  sample a target pose related to the terrain, not blind empty space.
"""

from __future__ import annotations

import math

from isaaclab.envs.mdp import UniformPoseCommandCfg
from isaaclab.utils import configclass

from mars_rover.mdp.kinematics import CHASSIS_BODY_NAME


@configclass
class CommandsCfg:
    """Command terms exposed to obs/reward code as command_name='target_pose'."""

    # NOTE:
    # UniformPoseCommandCfg is a solid Phase A starter: samples poses in ranges.
    # As you harden the stack, prefer a terrain-based sampler (RLRoverLab style)
    # so goals land on walkable surfaces.
    target_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name=CHASSIS_BODY_NAME,
        resampling_time_range=(150.0, 150.0),  # hold one goal for the episode window
        debug_vis=True,  # shows target marker in viewport — helpful while learning
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(3.0, 12.0),   # meters ahead-ish relative sampling ranges
            pos_y=(-6.0, 6.0),
            pos_z=(0.0, 0.0),    # keep on ground plane logic / yaw focus
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-math.pi, math.pi),
        ),
    )
