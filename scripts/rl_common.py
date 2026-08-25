# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared by train / play / evaluate (import AFTER AppLauncher)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import gymnasium as gym


def load_entry_cfg(entry_point: str):
    """Load ``module:Class`` and instantiate it."""
    module_name, attr = entry_point.split(":")
    return getattr(importlib.import_module(module_name), attr)()


def load_task_cfgs(task: str):
    """Return ``(env_cfg, agent_cfg)`` from the Gym spec kwargs."""
    spec = gym.spec(task)
    env_cfg = load_entry_cfg(spec.kwargs["env_cfg_entry_point"])
    agent_cfg = load_entry_cfg(spec.kwargs["rsl_rl_cfg_entry_point"])
    return env_cfg, agent_cfg


def write_run_manifest(log_dir: str | Path, *, seed: int, task: str, extra: dict | None = None) -> None:
    """Write experiment knobs next to checkpoints (not a science metric dump)."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "seed": seed,
        "obs_dim": 19,
        "action_dim": 6,
        "isaac_sim": "6.0.1",
        "isaac_lab": "v3.0.0-beta2.patch1",
    }
    if extra:
        payload.update(extra)
    (path / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
