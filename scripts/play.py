"""Load a Baseline v1 checkpoint and run the policy (visual / qualitative).

    ./isaaclab.sh -p /path/to/Mars_rover_rl/scripts/play.py \\
        --task Mars-Perseverance-Nav-v0 --checkpoint /path/to/model.pt --num_envs 1
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

parser = argparse.ArgumentParser(description="Play a trained Perseverance policy.")
parser.add_argument("--task", type=str, default="Mars-Perseverance-Nav-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import mars_rover.envs  # noqa: F401

from rl_common import load_task_cfgs


def main() -> None:
    env_cfg, agent_cfg = load_task_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    log_dir = Path("logs") / "rsl_rl" / "play"
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=env.unwrapped.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    print(f"[play] checkpoint={args_cli.checkpoint}")
    for step in range(args_cli.num_steps):
        actions = policy(obs)
        obs, _rewards, _dones, _extras = env.step(actions)
        if step % 50 == 0:
            print(f"[play] step={step}")
    env.close()
    print("[play] done")


if __name__ == "__main__":
    main()
    simulation_app.close()
