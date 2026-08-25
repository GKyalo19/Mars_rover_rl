"""Reset / randomization events for navigation episodes."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from mars_rover.mdp.events import reset_episode_buffers, validate_wheel_joints


@configclass
class EventCfg:
    """What happens at startup and when an episode resets."""

    validate_wheels = EventTerm(
        func=validate_wheel_joints,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reset_progress_idle = EventTerm(
        func=reset_episode_buffers,
        mode="reset",
        params={},
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
