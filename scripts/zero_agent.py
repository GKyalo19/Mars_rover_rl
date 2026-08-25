"""Step Mars-Perseverance-Nav-v0 with zero actions (wiring test).

Run on NVIDIA with Isaac Lab's Python, after:
  - pip install -e source/mars_rover_rl
  - Perseverance USD installed (see README)

Example:
  ./isaaclab.sh -p /path/to/Mars_rover_rl/scripts/zero_agent.py --num_envs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

parser = argparse.ArgumentParser(description="Zero-action smoke test for Perseverance nav.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Mars-Perseverance-Nav-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import mars_rover.envs  # noqa: F401

from rl_common import load_task_cfgs


def main() -> None:
    env_cfg, _agent_cfg = load_task_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, info = env.reset()
    print("[zero_agent] reset OK")

    action_dim = env.unwrapped.action_manager.total_action_dim
    if action_dim != 6:
        raise RuntimeError(f"Baseline v1 expects action_dim=6, got {action_dim}")

    obs_policy = obs["policy"] if isinstance(obs, dict) else obs
    obs_dim = obs_policy.shape[-1]
    if obs_dim != 19:
        raise RuntimeError(f"Baseline v1 expects obs_dim=19, got {obs_dim}")

    print(f"[zero_agent] action_dim={action_dim} obs_dim={obs_dim}")

    for step in range(100):
        actions = torch.zeros(env.unwrapped.num_envs, 6, device=env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        if step % 20 == 0:
            print(f"step={step} reward_mean={reward.mean().item():.4f}")

    env.close()
    print("[zero_agent] done — if you saw rewards printing, managers are alive.")


if __name__ == "__main__":
    main()
    simulation_app.close()
