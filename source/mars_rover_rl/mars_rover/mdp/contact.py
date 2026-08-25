# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Chassis contact helpers (unfiltered ContactSensor → net force)."""

from __future__ import annotations

import torch
from isaaclab.sensors import ContactSensor

from mars_rover.mdp.sim_data import as_torch


def max_net_contact_force(contact_sensor: ContactSensor) -> torch.Tensor:
    """Peak |net force| over history and bodies. Shape ``(N,)``.

    Use this when ``filter_prim_paths_expr`` is empty: ``force_matrix_w`` is
    not populated; ``net_forces_w_history`` is.
    """
    forces = as_torch(contact_sensor.data.net_forces_w_history)
    magnitude = torch.linalg.norm(forces, dim=-1)
    if magnitude.ndim == 3:
        return magnitude.amax(dim=(1, 2))
    return magnitude.amax(dim=1)
