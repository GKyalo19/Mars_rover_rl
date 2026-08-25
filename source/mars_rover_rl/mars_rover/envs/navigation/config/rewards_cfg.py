"""Reward weights / term wiring for Perseverance Baseline v1.

Conservative subset for the first PPO run. Extra terms exist as functions
and can be enabled later by adding a RewTerm with a non-zero weight.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from mars_rover.mdp import rewards as rewards
from mars_rover.mdp.kinematics import CHASSIS_CONTACT_THRESHOLD, UNSAFE_ATTITUDE_THRESHOLD


@configclass
class RewardsCfg:
    """Baseline v1: navigation + body safety + light time/idle shaping."""

    progress_to_goal = RewTerm(
        func=rewards.progress_to_goal,
        weight=5.0,
        params={"command_name": "target_pose"},
    )
    reached_goal = RewTerm(
        func=rewards.reached_goal,
        weight=10.0,
        params={
            "command_name": "target_pose",
            "distance_threshold": 0.25,
        },
    )
    collision = RewTerm(
        func=rewards.collision_penalty,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor"),
            "threshold": CHASSIS_CONTACT_THRESHOLD,
        },
    )
    safety_attitude = RewTerm(
        func=rewards.safety_attitude,
        weight=-5.0,
        params={"threshold": UNSAFE_ATTITUDE_THRESHOLD},
    )
    idle = RewTerm(
        func=rewards.idle_penalty,
        weight=-0.2,
        params={"speed_threshold": 0.05},
    )
    time_penalty = RewTerm(
        func=rewards.time_penalty,
        weight=-0.1,
    )
