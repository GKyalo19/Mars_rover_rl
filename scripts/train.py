"""Train Mars-Perseverance-Nav-v0 with RSL-RL PPO (Baseline v1).

Run on NVIDIA with Isaac Lab's Python after ``pip install -e source/mars_rover_rl``::

    ./isaaclab.sh -p /path/to/Mars_rover_rl/scripts/train.py \\
        --task Mars-Perseverance-Nav-v0 --num_envs 4 --max_iterations 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

parser = argparse.ArgumentParser(description="Train Perseverance Baseline v1 with RSL-RL.")
parser.add_argument("--task", type=str, default="Mars-Perseverance-Nav-v0")
parser.add_argument("--num_envs", type=int, default=4, help="Smoke default; raise after Gates A–D.")
parser.add_argument("--max_iterations", type=int, default=5, help="Smoke default; use 2000 for a real run.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import mars_rover.envs  # noqa: F401  — registers Gym ids

from rl_common import load_task_cfgs, write_run_manifest
from mars_rover.assets.robots.perseverance.perseverance import perseverance_usd_path
from mars_rover.assets.terrains.mars.terrain_spec import FLAT_TERRAIN_SPEC, ROUGH_TERRAIN_SPEC
from mars_rover.mdp.kinematics import (
    CHASSIS_BODY_NAME,
    CHASSIS_CONTACT_THRESHOLD,
    ROVER_WHEEL_JOINTS,
    WHEEL_ACTION_SCALE,
    WHEEL_DIRECTION_SIGNS,
    WHEEL_VEL_LIMIT,
)


def main() -> None:
    env_cfg, agent_cfg = load_task_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.resume = args_cli.resume

    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "params").mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=env.unwrapped.device)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    terrain_spec = ROUGH_TERRAIN_SPEC if "Rough" in args_cli.task else FLAT_TERRAIN_SPEC
    write_run_manifest(
        log_dir,
        seed=args_cli.seed,
        task=args_cli.task,
        extra={
            "num_envs": args_cli.num_envs,
            "max_iterations": args_cli.max_iterations,
            "experiment_name": agent_cfg.experiment_name,
            "terrain": terrain_spec.metadata(),
            "usd_path": perseverance_usd_path(),
            "wheel_joint_names": list(ROVER_WHEEL_JOINTS),
            "chassis_body_name": CHASSIS_BODY_NAME,
            "action_scale": WHEEL_ACTION_SCALE,
            "wheel_velocity_limit": WHEEL_VEL_LIMIT,
            "wheel_direction_signs": list(WHEEL_DIRECTION_SIGNS),
            "actuator_effort_limit_sim": 80.0,
            "decimation": env_cfg.decimation,
            "sim_dt": env_cfg.sim.dt,
            "episode_length_s": env_cfg.episode_length_s,
            "chassis_contact_threshold_n": CHASSIS_CONTACT_THRESHOLD,
            "reward_weights": {
                "progress_to_goal": env_cfg.rewards.progress_to_goal.weight,
                "reached_goal": env_cfg.rewards.reached_goal.weight,
                "collision": env_cfg.rewards.collision.weight,
                "safety_attitude": env_cfg.rewards.safety_attitude.weight,
                "idle": env_cfg.rewards.idle.weight,
                "time_penalty": env_cfg.rewards.time_penalty.weight,
            },
        },
    )
    (log_dir / "terrain_seed.txt").write_text(f"{terrain_spec.seed}\n")
    (log_dir / "terrain_metadata.json").write_text(json.dumps(terrain_spec.metadata(), indent=2) + "\n")

    print(f"[train] log_dir={log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    print("[train] done")


if __name__ == "__main__":
    main()
    simulation_app.close()
