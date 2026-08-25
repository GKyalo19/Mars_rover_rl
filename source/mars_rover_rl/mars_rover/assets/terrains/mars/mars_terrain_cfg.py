"""Procedural Mars-like terrain settings (Baseline v1).

Flat plane is the default env. Rough/rocks is a second Gym id.
Both are seeded so experiments are reproducible.

Requires Isaac Lab (NVIDIA) to import and spawn.
"""

from __future__ import annotations

from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import hf_terrains_cfg
from isaaclab.utils import configclass

from mars_rover.assets.terrains.mars.terrain_spec import FLAT_TERRAIN_SPEC, ROUGH_TERRAIN_SPEC


@configclass
class MarsFlatTerrainCfg(TerrainImporterCfg):
    """Stage-0 plane used for actuator, reset, and first PPO tests."""

    prim_path: str = "/World/ground"
    terrain_type: str = "plane"
    terrain_generator: TerrainGeneratorCfg | None = None
    debug_vis: bool = False


@configclass
class MarsProceduralTerrainCfg(TerrainImporterCfg):
    """Seeded height-field with discrete rock-like obstacles."""

    prim_path: str = "/World/ground"
    terrain_type: str = "generator"
    terrain_generator: TerrainGeneratorCfg = TerrainGeneratorCfg(
        seed=ROUGH_TERRAIN_SPEC.seed,
        size=ROUGH_TERRAIN_SPEC.size,
        border_width=3.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.05,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "rocks": hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg(
                proportion=1.0,
                obstacle_width_range=(0.4, 1.2),
                obstacle_height_range=(0.3, 1.0),
                num_obstacles=ROUGH_TERRAIN_SPEC.rock_count,
                platform_width=4.0,
                obstacle_height_mode="choice",
            )
        },
    )
    debug_vis: bool = False


# Metadata for experiment logs (not used by PhysX).
FLAT_TERRAIN_METADATA = FLAT_TERRAIN_SPEC.metadata()
ROUGH_TERRAIN_METADATA = ROUGH_TERRAIN_SPEC.metadata()
