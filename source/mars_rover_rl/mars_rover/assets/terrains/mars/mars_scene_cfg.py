"""Interactive scene: flat (default) or procedural Mars ground + Perseverance.

Height scan / cameras are deferred from Baseline v1 (add a perception scene later).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from mars_rover.assets.robots.perseverance.perseverance import PERSEVERANCE_CFG
from mars_rover.assets.terrains.mars.mars_terrain_cfg import MarsFlatTerrainCfg
from mars_rover.mdp.kinematics import chassis_prim_path


@configclass
class MarsNavSceneCfg(InteractiveSceneCfg):
    """One 'Mars yard' cloned across parallel environments."""

    terrain = MarsFlatTerrainCfg()

    robot = PERSEVERANCE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Body-only contacts: wheel-ground is expected and is not a crash.
    contact_sensor = ContactSensorCfg(
        prim_path=chassis_prim_path(),
        history_length=3,
        track_air_time=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.95, 0.9, 0.85)),
    )
