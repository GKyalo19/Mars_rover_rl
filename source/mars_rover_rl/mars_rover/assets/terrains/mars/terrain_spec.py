# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Thin terrain specification + spawn/goal metadata.

This is not a terrain engine. It holds the knobs the generator/config needs
and records valid task locations so the env does not infer them from meshes.
Isaac Lab importer configs live in ``mars_terrain_cfg.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerrainSpec:
    seed: int = 42
    size: tuple[float, float] = (40.0, 40.0)
    roughness: float = 0.0
    rock_count: int = 0
    difficulty: int = 0

    def metadata(self) -> dict:
        """Generated for reproducibility, logging, and future spawn/goal validation.

        The env does not yet constrain commands from these points.
        ``roughness`` is recorded but not yet wired into the height-field generator.
        """
        return {
            "seed": self.seed,
            "size": self.size,
            "roughness": self.roughness,
            "rock_count": self.rock_count,
            "difficulty": self.difficulty,
            "spawn_points": [(0.0, 0.0)],
            # Matches CommandsCfg target ranges (ahead / to the sides).
            "goal_points": [
                (3.0, 0.0),
                (12.0, 0.0),
                (7.5, 6.0),
                (7.5, -6.0),
            ],
        }


FLAT_TERRAIN_SPEC = TerrainSpec(seed=42, roughness=0.0, rock_count=0, difficulty=0)
ROUGH_TERRAIN_SPEC = TerrainSpec(seed=42, roughness=0.5, rock_count=60, difficulty=2)
