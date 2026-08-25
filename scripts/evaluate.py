"""Quantitative evaluation — physical/task metrics, not reward.

Captures termination reason and pose **inside** ``_reset_idx`` so Isaac Lab's
automatic reset cannot overwrite the terminal state.

    ./isaaclab.sh -p scripts/evaluate.py --checkpoint <path> --num_episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

parser = argparse.ArgumentParser(description="Evaluate Perseverance Baseline v1 metrics.")
parser.add_argument("--task", type=str, default="Mars-Perseverance-Nav-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=100_000, help="Safety cutoff.")
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import mars_rover.envs  # noqa: F401

from mars_rover.mdp.sim_data import as_torch
from rl_common import load_task_cfgs


class _ResetCapture:
    env_ids = None
    xy = None
    goal = None
    collision = None
    lengths = None


def _install_reset_capture(unwrapped, capture: _ResetCapture):
    orig = unwrapped._reset_idx

    def _reset_idx(env_ids):
        if env_ids is not None:
            ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=unwrapped.device)
            if ids.numel() > 0:
                robot = unwrapped.scene["robot"]
                capture.xy = as_torch(robot.data.root_pos_w)[:, :2].clone()
                tm = unwrapped.termination_manager
                capture.goal = tm.get_term("goal_reached").clone()
                capture.collision = tm.get_term("collision").clone()
                capture.lengths = unwrapped.episode_length_buf.clone()
                capture.env_ids = ids.clone()
        return orig(env_ids)

    unwrapped._reset_idx = _reset_idx


def main() -> None:
    env_cfg, agent_cfg = load_task_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    log_dir = Path("logs") / "rsl_rl" / "eval"
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device=env.unwrapped.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    capture = _ResetCapture()
    _install_reset_capture(unwrapped, capture)

    robot = unwrapped.scene["robot"]
    dt = unwrapped.step_dt
    n = unwrapped.num_envs
    device = unwrapped.device

    successes = 0
    collisions = 0
    episodes = 0
    distance_sum = 0.0
    time_to_goal_sum = 0.0
    time_to_goal_count = 0
    length_sum = 0.0

    ep_distance = torch.zeros(n, device=device)
    xy_before = as_torch(robot.data.root_pos_w)[:, :2].clone()

    obs = wrapped.get_observations()
    for _ in range(args_cli.max_steps):
        if episodes >= args_cli.num_episodes:
            break
        capture.env_ids = None
        actions = policy(obs)
        obs, _rewards, dones, _extras = wrapped.step(actions)
        xy_after = as_torch(robot.data.root_pos_w)[:, :2]
        done = dones.bool()

        if capture.env_ids is not None:
            for i in capture.env_ids.tolist():
                if episodes >= args_cli.num_episodes:
                    break
                last_move = torch.norm(capture.xy[i] - xy_before[i], p=2)
                dist = float(ep_distance[i] + last_move)
                length_s = float(capture.lengths[i]) * dt
                episodes += 1
                distance_sum += dist
                length_sum += length_s
                if bool(capture.goal[i]):
                    successes += 1
                    time_to_goal_sum += length_s
                    time_to_goal_count += 1
                if bool(capture.collision[i]):
                    collisions += 1
                ep_distance[i] = 0.0
            alive = ~done
            if alive.any():
                ep_distance[alive] = ep_distance[alive] + torch.norm(xy_after[alive] - xy_before[alive], p=2, dim=-1)
        else:
            ep_distance = ep_distance + torch.norm(xy_after - xy_before, p=2, dim=-1)

        xy_before = xy_after.clone()

    wrapped.close()

    metrics = {
        "episodes": episodes,
        "success_rate": (successes / episodes) if episodes else 0.0,
        "mean_distance_m": (distance_sum / episodes) if episodes else 0.0,
        "mean_time_to_goal_s": (time_to_goal_sum / time_to_goal_count) if time_to_goal_count else None,
        "collision_rate": (collisions / episodes) if episodes else 0.0,
        "mean_episode_s": (length_sum / episodes) if episodes else 0.0,
        "checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
    }
    print("[evaluate] task metrics (not reward):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    if args_cli.out:
        out = Path(args_cli.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
