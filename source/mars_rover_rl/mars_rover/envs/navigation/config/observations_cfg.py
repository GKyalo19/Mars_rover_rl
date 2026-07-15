# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation wiring diagram for Perseverance navigation (Phase A).

This is NOT the neural network. It tells Isaac Lab's ObservationManager:
which functions to call, in what order, and how to scale them.

Requires Isaac Lab (run / import on NVIDIA).
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import mars_rover.envs.navigation.mdp as mdp


@configclass
class ObservationsCfg:
    """All observation groups for the task.

    Puzzle piece: PPO reads the ``policy`` group. Later you could add a
    separate ``critic`` group with privileged info — we keep one group for now.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Vector observation fed to the actor (and usually the critic too)."""

        # Order matters when concatenate_terms=True — it becomes one flat vector.
        distance_to_goal = ObsTerm(
            func=mdp.observations.distance_to_goal,
            params={"command_name": "target_pose"},
            scale=0.1,  # shrink meters so values aren't huge vs angles
        )
        heading_to_goal = ObsTerm(
            func=mdp.observations.heading_to_goal,
            params={"command_name": "target_pose"},
            scale=1.0 / math.pi,  # map radians roughly into [-1, 1]
        )
        last_action = ObsTerm(func=mdp.observations.last_action)

        # height_scan / base velocities land in a later pass once the scene
        # sensors and articulation cfg exist (still Phase A, just more terms).

        def __post_init__(self) -> None:
            # True: stack terms into one tensor for a plain MLP policy.
            self.concatenate_terms = True
            # True later: add noise for domain randomization / robustness.
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
