# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Differential-drive ActionTerm: policy [v, ω] → six wheel velocity targets.

Requires Isaac Lab at runtime (NVIDIA). Math is delegated to
``mars_rover.mdp.kinematics`` so the gearbox stays unit-testable on a Mac.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from mars_rover.mdp.kinematics import WHEEL_ORDER, twist_to_wheel_velocities

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class DifferentialDriveAction(ActionTerm):
    """Skid-steer action term for a 6-wheel rocker-bogie style rover (Phase A).

    Lifecycle each control step (Isaac Lab action manager):

    1. ``process_actions(actions)`` — scale / clip policy output, cache it
    2. ``apply_actions()`` — convert twist → wheel ω, write joint velocity targets

    Puzzle piece: the neural net never talks to PhysX joints directly. It speaks
    chassis language; this class is the translator.
    """

    cfg: actions_cfg.DifferentialDriveActionCfg
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.DifferentialDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.wheel_joint_names)

        if len(self._joint_ids) != 6:
            raise RuntimeError(
                f"Expected 6 wheel joints matching {cfg.wheel_joint_names}, "
                f"found {len(self._joint_ids)}: {self._joint_names}. "
                "Fix USD joint names or wheel_joint_names in the cfg."
            )

        # Buffers: one row per parallel env.
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._scale = torch.tensor(cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        """Policy output size: [linear_x, angular_z]."""
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Cache policy actions and map into physical [v, ω] with clips."""
        self._raw_actions[:] = actions
        scaled = self._raw_actions * self._scale + self._offset
        # Clamp to safe chassis limits (industry: never trust unbounded NN outputs).
        scaled[:, 0].clamp_(-self.cfg.max_linear_vel, self.cfg.max_linear_vel)
        scaled[:, 1].clamp_(-self.cfg.max_angular_vel, self.cfg.max_angular_vel)
        self._processed_actions[:] = scaled

    def apply_actions(self) -> None:
        """Write velocity targets to the six wheel revolute joints."""
        v = self._processed_actions[:, 0]
        w = self._processed_actions[:, 1]
        wheel_vel = twist_to_wheel_velocities(
            v,
            w,
            track_width=self.cfg.track_width,
            wheel_radius=self.cfg.wheel_radius,
        )
        # wheel_vel columns follow WHEEL_ORDER; joint_ids follow find_joints order.
        # Reorder targets to match whatever order find_joints returned.
        name_to_col = {name: i for i, name in enumerate(WHEEL_ORDER)}
        # Our cfg joint names are like "wheel_FL_joint" → token "FL"
        ordered = []
        for joint_name in self._joint_names:
            token = None
            for key in WHEEL_ORDER:
                if key in joint_name:
                    token = key
                    break
            if token is None:
                raise RuntimeError(f"Cannot map joint '{joint_name}' to {WHEEL_ORDER}")
            ordered.append(wheel_vel[:, name_to_col[token]])
        targets = torch.stack(ordered, dim=-1)
        self._asset.set_joint_velocity_target(targets, joint_ids=self._joint_ids)
