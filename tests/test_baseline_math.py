# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Mac-side tests for Baseline v1 geometry, action mapping, and reward math."""

from __future__ import annotations

import math

import torch

from mars_rover.assets.terrains.mars.terrain_spec import FLAT_TERRAIN_SPEC, ROUGH_TERRAIN_SPEC
from mars_rover.mdp.kinematics import (
    CHASSIS_BODY_NAME,
    ROVER_WHEEL_JOINTS,
    UNSAFE_ATTITUDE_THRESHOLD,
    WHEEL_ACTION_SCALE,
    WHEEL_ORDER,
    WHEEL_VEL_LIMIT,
    action_index_to_joint,
    apply_wheel_direction_signs,
    chassis_prim_path,
    roll_pitch_from_quat_xyzw,
    scale_and_clip_wheel_actions,
    wheel_action_scale_map,
    xy_world_to_base_horizontal,
    yaw_from_quat_xyzw,
)
from mars_rover.mdp.reward_math import (
    actuation_effort_from_wheel_vel,
    attitude_unsafe_mask,
    idle_from_consecutive_steps,
    idle_from_speed,
    temporal_progress,
)
from mars_rover.mdp.sim_data import as_torch


def _quat_from_yaw_xyzw(yaw: float) -> torch.Tensor:
    """Isaac Lab 3.0 convention: ``(x, y, z, w)`` for a yaw about world Z."""
    half = yaw * 0.5
    return torch.tensor([[0.0, 0.0, math.sin(half), math.cos(half)]])


def test_scale_and_clip_wheel_actions():
    raw = torch.tensor([[1.0, -1.0, 2.0, 0.0, 0.5, -2.0]])
    out = scale_and_clip_wheel_actions(raw)
    assert out.shape == (1, 6)
    assert torch.allclose(out[0, 0], torch.tensor(WHEEL_ACTION_SCALE))
    assert torch.allclose(out[0, 1], torch.tensor(-WHEEL_ACTION_SCALE))
    assert torch.all(out.abs() <= WHEEL_VEL_LIMIT + 1e-6)


def test_action_index_maps_to_usd_joint_names():
    contract = action_index_to_joint()
    assert [c[0] for c in contract] == list(range(6))
    assert [c[1] for c in contract] == list(WHEEL_ORDER)
    assert [c[2] for c in contract] == [
        "Joint_Wheel_FL",
        "Joint_Wheel_ML",
        "Joint_Wheel_RL",
        "Joint_Wheel_FR",
        "Joint_Wheel_MR",
        "Joint_Wheel_RR",
    ]
    scale_map = wheel_action_scale_map()
    assert scale_map["Joint_Wheel_FL"] == WHEEL_ACTION_SCALE
    assert apply_wheel_direction_signs(torch.ones(1, 6)).shape == (1, 6)


def test_chassis_body_and_prim_path():
    assert CHASSIS_BODY_NAME == "Body"
    assert chassis_prim_path() == "{ENV_REGEX_NS}/Robot/Rover/Body"


def test_as_torch_passthrough_plain_tensor():
    t = torch.tensor([1.0, 2.0])
    assert as_torch(t) is t


def test_goal_in_front_is_positive_dx():
    robot = torch.tensor([[0.0, 0.0]])
    goal = torch.tensor([[5.0, 0.0]])
    yaw = torch.tensor([0.0])
    xy = xy_world_to_base_horizontal(goal, robot, yaw)
    assert torch.allclose(xy, torch.tensor([[5.0, 0.0]]), atol=1e-5)


def test_goal_to_the_left_is_positive_dy():
    robot = torch.tensor([[0.0, 0.0]])
    goal = torch.tensor([[0.0, 4.0]])
    yaw = torch.tensor([0.0])
    xy = xy_world_to_base_horizontal(goal, robot, yaw)
    assert torch.allclose(xy, torch.tensor([[0.0, 4.0]]), atol=1e-5)


def test_yaw_rotates_world_goal_into_body_frame():
    robot = torch.tensor([[0.0, 0.0]])
    goal = torch.tensor([[0.0, 3.0]])
    yaw = torch.tensor([math.pi / 2])
    xy = xy_world_to_base_horizontal(goal, robot, yaw)
    assert torch.allclose(xy, torch.tensor([[3.0, 0.0]]), atol=1e-5)


def test_yaw_from_quat_matches_constructor():
    yaw = torch.tensor([0.3])
    got = yaw_from_quat_xyzw(_quat_from_yaw_xyzw(0.3))
    assert torch.allclose(got, yaw, atol=1e-5)


def test_xyzw_identity_is_level_and_unyawed():
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(yaw_from_quat_xyzw(identity), torch.zeros(1), atol=1e-5)
    assert torch.allclose(roll_pitch_from_quat_xyzw(identity), torch.zeros(1, 2), atol=1e-5)


def test_old_wxyz_identity_is_not_lab3_identity():
    """Lab 2 identity ``(w, x, y, z) = (1, 0, 0, 0)`` is a 180° roll if read as XYZW."""
    wxyz_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rp = roll_pitch_from_quat_xyzw(wxyz_identity)
    assert not torch.allclose(rp, torch.zeros(1, 2), atol=0.1)


def test_level_quat_has_zero_roll_pitch():
    rp = roll_pitch_from_quat_xyzw(_quat_from_yaw_xyzw(0.7))
    assert torch.allclose(rp, torch.zeros(1, 2), atol=1e-5)


def test_roll_from_xyzw_quat():
    angle = 0.4
    half = angle * 0.5
    quat = torch.tensor([[math.sin(half), 0.0, 0.0, math.cos(half)]])
    rp = roll_pitch_from_quat_xyzw(quat)
    assert torch.allclose(rp[0, 0], torch.tensor(angle), atol=1e-5)
    assert torch.allclose(rp[0, 1], torch.tensor(0.0), atol=1e-5)


def test_temporal_progress_is_previous_minus_current():
    prev = torch.tensor([10.0, 4.0])
    current = torch.tensor([9.0, 5.0])
    initialized = torch.tensor([True, True])
    out = temporal_progress(prev, current, initialized)
    assert torch.allclose(out, torch.tensor([1.0, -1.0]))
    first = temporal_progress(prev, current, torch.tensor([False, False]))
    assert torch.allclose(first, torch.zeros(2))


def test_idle_from_speed():
    speed = torch.tensor([0.0, 0.2])
    out = idle_from_speed(speed, threshold=0.05, max_episode_length=10.0)
    assert torch.allclose(out[0], torch.tensor(0.1))
    assert torch.allclose(out[1], torch.tensor(0.0))


def test_idle_from_consecutive_steps():
    steps = torch.tensor([5, 30, 31])
    out = idle_from_consecutive_steps(steps, min_steps=30, max_episode_length=10.0)
    assert torch.allclose(out[0], torch.tensor(0.0))
    assert torch.allclose(out[1], torch.tensor(0.1))
    assert torch.allclose(out[2], torch.tensor(0.1))


def test_attitude_threshold_is_configurable():
    rp = torch.tensor([[0.8, 0.0], [0.1, 0.1]])
    mask = attitude_unsafe_mask(rp, UNSAFE_ATTITUDE_THRESHOLD)
    assert bool(mask[0])
    assert not bool(mask[1])
    assert not bool(attitude_unsafe_mask(rp, threshold=0.9)[0])


def test_actuation_effort_is_mean_abs_wheel_speed():
    vel = torch.tensor([[1.0, -1.0, 0.0, 0.0, 2.0, -2.0]])
    assert torch.allclose(actuation_effort_from_wheel_vel(vel), torch.tensor([1.0]))


def test_terrain_spec_metadata_is_thin():
    meta = FLAT_TERRAIN_SPEC.metadata()
    assert meta["difficulty"] == 0
    assert meta["rock_count"] == 0
    assert "spawn_points" in meta
    assert "goal_points" in meta
    rough = ROUGH_TERRAIN_SPEC.metadata()
    assert rough["rock_count"] == 60
    assert rough["seed"] == 42


def test_canonical_joint_list_is_the_single_contract():
    assert list(ROVER_WHEEL_JOINTS) == [
        "Joint_Wheel_FL",
        "Joint_Wheel_ML",
        "Joint_Wheel_RL",
        "Joint_Wheel_FR",
        "Joint_Wheel_MR",
        "Joint_Wheel_RR",
    ]
