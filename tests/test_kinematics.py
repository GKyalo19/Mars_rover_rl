# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for skid-steer kinematics (no Isaac Lab required)."""

from __future__ import annotations

import torch

from mars_rover.mdp.kinematics import WHEEL_ORDER, twist_to_wheel_velocities


def test_wheel_order_has_six_names():
    assert len(WHEEL_ORDER) == 6


def test_straight_line_equal_wheel_speeds():
    """Pure forward v, zero yaw → left and right wheels match."""
    v = torch.tensor([1.0])
    w = torch.tensor([0.0])
    out = twist_to_wheel_velocities(v, w, track_width=2.0, wheel_radius=0.5)
    # v/r = 1/0.5 = 2 rad/s on every wheel
    assert out.shape == (1, 6)
    assert torch.allclose(out, torch.full((1, 6), 2.0))


def test_spin_in_place_opposite_sides():
    """Zero forward, positive yaw → left negative, right positive."""
    v = torch.tensor([0.0])
    w = torch.tensor([1.0])  # rad/s
    B = 2.0
    r = 0.5
    out = twist_to_wheel_velocities(v, w, track_width=B, wheel_radius=r)
    # v_L = 0 - 1*(B/2) = -1 → ω_L = -2; v_R = +1 → ω_R = +2
    expected_left = -1.0 / r
    expected_right = 1.0 / r
    assert torch.allclose(out[0, 0], torch.tensor(expected_left))
    assert torch.allclose(out[0, 2], torch.tensor(expected_left))
    assert torch.allclose(out[0, 4], torch.tensor(expected_left))
    assert torch.allclose(out[0, 1], torch.tensor(expected_right))
    assert torch.allclose(out[0, 3], torch.tensor(expected_right))
    assert torch.allclose(out[0, 5], torch.tensor(expected_right))


def test_batch_envs():
    v = torch.tensor([0.5, 1.0, 0.0])
    w = torch.tensor([0.0, 0.0, 0.2])
    out = twist_to_wheel_velocities(v, w, track_width=2.0, wheel_radius=0.25)
    assert out.shape == (3, 6)
