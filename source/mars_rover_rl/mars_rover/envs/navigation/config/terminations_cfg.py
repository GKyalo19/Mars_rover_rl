"""Episode end conditions for Perseverance navigation."""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from mars_rover.mdp import terminations as terminations
from mars_rover.mdp.kinematics import CHASSIS_CONTACT_THRESHOLD, UNSAFE_ATTITUDE_THRESHOLD


@configclass
class TerminationsCfg:
    """When any term returns True for an env, that env resets."""

    time_out = DoneTerm(func=terminations.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=terminations.goal_reached,
        params={"command_name": "target_pose", "distance_threshold": 0.25},
    )
    collision = DoneTerm(
        func=terminations.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": CHASSIS_CONTACT_THRESHOLD},
    )
    out_of_bounds = DoneTerm(
        func=terminations.farther_than_allowed,
        params={"command_name": "target_pose", "max_distance": 25.0},
    )
    unsafe_attitude = DoneTerm(
        func=terminations.unsafe_attitude,
        params={"threshold": UNSAFE_ATTITUDE_THRESHOLD},
    )
