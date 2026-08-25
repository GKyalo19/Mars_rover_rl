# Chapter 7 — PPO Training Loop (RSL-RL)

> **How we work**  
>
> - Explanations + full code live **in this chapter**.  
> - **You** create files and paste while reading.  
> - I will **not** write into your `source/` package unless you ask.
>
> **Goal:** Assemble the full navigation environment, register it with Gymnasium, configure PPO, and understand how to launch training with **RSL-RL** on the NVIDIA machine.  
> **Depends on:** Ch.2–6 (package, obs/actions, rewards/terminations, terrain/goals).  
> **Hard gate:** Training only runs for real once your Perseverance `ArticulationCfg` is no longer `MISSING` (Ch.3 / 3b USD + joints). You can still paste and understand everything now.  
> **Next:** Ch.8 vision / reach-avoid · Ch.9 evaluation.

---



## 7.0 Bigger picture — the student finally enters the classroom

All previous chapters built **classroom infrastructure**:


| Chapter | Built                                        |
| ------- | -------------------------------------------- |
| 2       | Importable library                           |
| 3 / 3b  | Physical rover (joints, drives)              |
| 4       | Senses + intents (`o`, `[v,ω]`)              |
| 5       | Gradebook + end-of-exam bell                 |
| 6       | Mars yard + destinations                     |
| **7**   | **The student (PPO) who studies by driving** |


```text
gym.make("Mars-Perseverance-Nav-v0")
        │
        ▼
ManagerBasedRLEnv  (scene + managers from Ch.4–6)
        │
        ▼
RslRlVecEnvWrapper  (translate Lab tensors ↔ RSL-RL API)
        │
        ▼
OnPolicyRunner / PPO
  collect rollouts → compute advantages → clipped update → repeat
```



### Industry parallel

This is the standard **“env factory + trainer”** pattern used across robot learning labs:

1. Register a task id (`Mars-Perseverance-Nav-v0`)
2. `gym.make(task)` builds the vectorized simulator
3. A library-specific **wrapper** adapts interfaces
4. A **runner** owns the learning loop and checkpoints

Isaac Lab already ships battle-tested `train.py` scripts under `scripts/reinforcement_learning/rsl_rl/`.  
In this chapter you will:

- build **your** env + agent configs  
- register the Gym id  
- understand (and optionally paste) a thin training entry script  
- know how to call Lab’s official trainer with `--task ...`

---



## 7.1 Jargon, explained properly



### Gymnasium / `gym.make` / registration

**Gymnasium** is the modern API (successor to OpenAI Gym) for saying:

> “Give me an environment with this id.”

**Registration** is how you advertise your task:

```python
gym.register(id="Mars-Perseverance-Nav-v0", entry_point=..., kwargs={...})
```

After registration (usually by importing your package), anyone can:

```python
env = gym.make("Mars-Perseverance-Nav-v0")
```

**Why industry does this:** task ids become a shared vocabulary in papers, scripts, and CI: “train `Mars-Perseverance-Nav-v0`,” not “run this random Python file with 12 hardcoded paths.”

### Entry point

An **entry point** is a string address of a class, like:

```text
"isaaclab.envs:ManagerBasedRLEnv"
```

Meaning: module `isaaclab.envs`, class `ManagerBasedRLEnv`.

Your env cfg is another entry point:

```text
"mars_rover.envs.navigation.config.navigation_env_cfg:NavigationEnvCfg"
```



### ManagerBasedRLEnv

This is Isaac Lab’s environment class that:

- owns the scene (terrain, robot, sensors)  
- runs observation / action / reward / termination / command / event **managers**  
- steps many envs in parallel on GPU

You rarely subclass it for Phase A. You mostly write **configs**; Lab builds the env from them.

### Vectorized environment

Instead of one rover, you may run **N** rovers (e.g. 64, 256, 1024) at once.

Each step returns tensors with a leading batch dimension `N`. That is why reward functions returned `Tensor[num_envs]` in Ch.5.

**Why:** sample efficiency / wall-clock speed. PPO wants lots of experience; GPUs are good at many physics clones.

### Wrapper (`RslRlVecEnvWrapper`)

Lab’s env API is tensor-based and Lab-specific. RSL-RL expects its own interface.

A **wrapper** sits between them and translates calls/returns — like a power adapter between plug shapes.  
Without it, you would rewrite either Lab or RSL-RL. With it, both stay clean.

### RSL-RL

**RSL-RL** (Rudin et al.’s library, widely used with Isaac/Orbit/Lab) implements high-performance **on-policy** algorithms, especially **PPO**, for robotics.

You already met PPO math in Chapter 1. Here PPO becomes a running program:

- collect on-policy rollouts  
- estimate advantages (GAE)  
- optimize clipped surrogate + value loss − entropy bonus  
- save checkpoints



### On-policy

**On-policy** means: learn from data collected by the **current** policy, then discard (or stop reusing) that data after updates.

Contrast **off-policy** (DQN, SAC): can learn from a replay buffer of older behavior.

PPO is on-policy → you continually gather fresh driving experience after each update.

### Rollout / iteration


| Term                 | Meaning                                                                        |
| -------------------- | ------------------------------------------------------------------------------ |
| **Step**             | One env control tick (action → physics → reward)                               |
| **Rollout**          | A stretch of steps collected before an update (per env: `num_steps_per_env`)   |
| **Iteration**        | One “collect + learn” cycle of the runner                                      |
| **Epoch** (learning) | How many times we reuse the same rollout batch for gradient updates inside PPO |
| **Mini-batch**       | Slice of the rollout used for one optimizer step                               |


Example intuition:

- 128 envs  
- 24 steps per env per iteration  
- → 128 \times 24 = 3072 transitions collected per iteration



### Actor–Critic (again, practical)


| Network            | Job in training                                                     |
| ------------------ | ------------------------------------------------------------------- |
| **Actor** (policy) | Outputs mean (and std) of `[v, ω]` distribution                     |
| **Critic** (value) | Predicts expected return from observation — baseline for advantages |


Both are MLPs for Phase A (vector observations). Vision policies come in Ch.8.

### Checkpoint

A **checkpoint** is a saved snapshot of network weights (and often optimizer/runner state) you can reload to:

- resume training  
- play / evaluate (Ch.9)



### Hyperparameters

Knobs that are **not** learned by gradient descent but chosen by you:

- learning rate  
- `gamma`, `lam` (GAE)  
- clip `epsilon`  
- network width  
- entropy coefficient

Bad hyperparameters can make a correct MDP look “unlearnable.” Tuning is part of the craft.

### Decimation

**Decimation** = how many physics substeps run per one RL action.

If `sim.dt = 1/60` and `decimation = 4`, the policy acts at 60/4 = 15 Hz.

Too fast acting → twitchy, costly. Too slow → sluggish control. Start with Lab-like values and adjust.

### Headless

**Headless** means running without opening the GUI viewport — faster for training on the NVIDIA box.

You still log metrics; you just do not render pretty windows every step.

---



## 7.2 PPO in one operational paragraph (tying Ch.1 to code)

Each iteration roughly:

1. For `num_steps_per_env` steps, in all envs:
  - observe o  
  - sample a \sim \pi_\theta(\cdot\mid o)  
  - step env → r, o', done  
  - store transition
2. Compute advantages \hat{A} with GAE (\gamma, \lambda).
3. For several epochs / mini-batches, maximize PPO’s **clipped** objective, fit the value function, keep some **entropy** for exploration.
4. Save logs; periodically checkpoint.
5. Repeat until `max_iterations`.

Your Ch.4–6 work decides whether those transitions contain a learnable skill. PPO only optimizes whatever reward you defined.

---



## 7.3 Files you will create (you paste)

```text
source/mars_rover_rl/mars_rover/envs/navigation/config/
    navigation_env_cfg.py      # assembles full ManagerBasedRLEnvCfg

source/mars_rover_rl/mars_rover/envs/navigation/
    __init__.py                # gym.register(...)  (extend what you have)

configs/agents/   OR   source/.../agents/
    rsl_rl_ppo_cfg.py          # PPO runner / network / algorithm

scripts/
    train.py                   # thin launcher (optional if using Lab's train.py)
    zero_agent.py              # smoke-test env with zero actions (recommended)
```

---



## 7.4 Assemble the env cfg — full code to copy



### File: `envs/navigation/config/navigation_env_cfg.py`

This is the **master blueprint** that imports your Ch.4–6 pieces.

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Full navigation env configuration for Perseverance on procedural Mars terrain."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from mars_rover.assets.terrains.mars.mars_scene_cfg import MarsNavSceneCfg
from mars_rover.envs.navigation.config.actions_cfg import ActionsCfg
from mars_rover.envs.navigation.config.commands_cfg import CommandsCfg
from mars_rover.envs.navigation.config.events_cfg import EventCfg
from mars_rover.envs.navigation.config.observations_cfg import ObservationsCfg
from mars_rover.envs.navigation.config.rewards_cfg import RewardsCfg
from mars_rover.envs.navigation.config.terminations_cfg import TerminationsCfg


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Perseverance navigation MDP.

    Before training works:
      1) MarsNavSceneCfg.robot must be a real ArticulationCfg (not MISSING).
      2) Joint names in ActionsCfg must match that articulation.
      3) Contact sensor prim paths should match your robot link names.
    """

    # Scene: terrain, sensors, lights, robot slot
    scene: MarsNavSceneCfg = MarsNavSceneCfg(num_envs=64, env_spacing=8.0)

    # MDP managers
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Timing and viewer defaults — tune once the robot drives stably."""
        # Control frequency: policy acts every (decimation * sim.dt) seconds.
        self.decimation = 4
        self.sim.dt = 1.0 / 60.0
        self.episode_length_s = 60.0  # wall-clock seconds per episode budget

        # Viewer (ignored when headless)
        self.viewer.eye = (12.0, 12.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        # Keep sensor update rates aligned with policy rate when present.
        if getattr(self.scene, "contact_sensor", None) is not None:
            self.scene.contact_sensor.update_period = self.sim.dt * self.decimation
        if getattr(self.scene, "height_scanner", None) is not None:
            self.scene.height_scanner.update_period = self.sim.dt * self.decimation
```



### How to read this file


| Block                      | Role                            |
| -------------------------- | ------------------------------- |
| `scene`                    | The Mars yard + rover + sensors |
| `observations` / `actions` | Policy I/O (Ch.4)               |
| `rewards` / `terminations` | Teaching + episode ends (Ch.5)  |
| `commands` / `events`      | Goals + resets (Ch.6)           |
| `__post_init__`            | Clock rates for sim vs policy   |


This is the puzzle frame that holds every previous piece.

---



## 7.5 Register the Gym task — full code to copy



### File: update `mars_rover/envs/navigation/__init__.py`

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation tasks package — registers Gym ids on import."""

import gymnasium as gym

gym.register(
    id="Mars-Perseverance-Nav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "mars_rover.envs.navigation.config.navigation_env_cfg:NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": "mars_rover.envs.navigation.agents.rsl_rl_ppo_cfg:PerseverancePPORunnerCfg",
    },
)
```

Also ensure your package import path loads this module (for example from `mars_rover/envs/__init__.py`):

```python
# mars_rover/envs/__init__.py
from . import navigation as navigation  # noqa: F401  # triggers gym.register
```

And that `pip install -e` is done in the **Isaac Lab Python** on NVIDIA.

### What each register field means


| Field                    | Meaning                                                             |
| ------------------------ | ------------------------------------------------------------------- |
| `id`                     | Public task name you pass to `--task`                               |
| `entry_point`            | Env class to construct                                              |
| `env_cfg_entry_point`    | Your MDP blueprint                                                  |
| `rsl_rl_cfg_entry_point` | Your PPO hyperparameters class                                      |
| `disable_env_checker`    | Lab envs are tensor/vectorized; Gym’s default checker is too strict |


---



## 7.6 PPO agent config — full code to copy



### File: `mars_rover/envs/navigation/agents/rsl_rl_ppo_cfg.py`

Create the `agents/` folder, add an empty `__init__.py`, then paste:

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO settings for Perseverance navigation.

These numbers are *starting points*. Expect to tune after you see TensorBoard curves.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PerseverancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # How many env steps to collect per env between updates
    num_steps_per_env = 24
    # How many collect+update cycles to run
    max_iterations = 2000
    # Save checkpoints every N iterations
    save_interval = 100
    # Folder name under logs/
    experiment_name = "perseverance_nav"
    # Run name optional; timestamp used if empty
    run_name = ""
    # Resume helpers
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # exploration noise at start
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,          # PPO epsilon
        entropy_coef=0.01,       # exploration encouragement
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,              # discount
        lam=0.95,                # GAE lambda
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```



### Hyperparameter meanings in rover English


| Knob                | If too low                    | If too high                                    |
| ------------------- | ----------------------------- | ---------------------------------------------- |
| `learning_rate`     | Learns glacially              | Unstable / collapses                           |
| `clip_param`        | Tiny updates, slow            | Large policy swings                            |
| `entropy_coef`      | Exploits early, may get stuck | Wiggles forever                                |
| `gamma`             | Myopic (only near rewards)    | Values distant future (needs stable bootstrap) |
| `num_steps_per_env` | Noisy advantages              | Stale on-policy data / more memory             |
| network width       | Underfits                     | Slower, may overfit noise                      |


For navigation with a modest vector obs, `[256, 128, 64]` is a reasonable first try — larger than cartpole toys, smaller than giant vision nets.

---



## 7.7 Zero-agent smoke test — full code to copy (do this before PPO)

Before blaming PPO, prove the env steps.

### File: `scripts/zero_agent.py`

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Step Mars-Perseverance-Nav-v0 with zero actions (wiring test).

Run on NVIDIA with Isaac Lab's Python, after:
  - pip install -e source/mars_rover_rl
  - robot ArticulationCfg is filled (not MISSING)

Example:
  ./isaaclab.sh -p /path/to/Mars_rover_rl/scripts/zero_agent.py --num_envs 4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action smoke test for Perseverance nav.")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# Importing the package should register the Gym id.
import mars_rover.envs  # noqa: F401


def main() -> None:
    env = gym.make("Mars-Perseverance-Nav-v0", num_envs=args_cli.num_envs)
    obs, info = env.reset()
    print("[zero_agent] reset OK")

    # Action dim 2: [v, ω] — send zeros = "don't drive"
    for step in range(100):
        actions = torch.zeros(env.unwrapped.num_envs, 2, device=env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        if step % 20 == 0:
            print(f"step={step} reward_mean={reward.mean().item():.4f}")

    env.close()
    print("[zero_agent] done — if you saw rewards printing, managers are alive.")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

**Pass criteria:** process starts, reset works, 100 steps print finite rewards, no instant crash loop for every env (some collisions possible if spawn overlaps rocks — then fix spawn/terrain).

---



## 7.8 Training launch options

### Option A — Use Isaac Lab’s official RSL-RL train script (recommended)

On the NVIDIA machine, from your Isaac Lab install:

```bash
# once
./isaaclab.sh -i rsl_rl

# ensure our package is installed into Lab's python
./isaaclab.sh -p -m pip install -e /path/to/Mars_rover_rl/source/mars_rover_rl

# train
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Mars-Perseverance-Nav-v0 \
  --headless \
  --num_envs 64 \
  --max_iterations 2000
```

### What each line actually does

`isaaclab.sh` is Isaac Lab's own launcher script — it wraps Lab's bundled Python/conda environment so you don't have to manually `source` or `conda activate` anything. Every command below runs *through* it rather than calling `python` directly.

| Command | Plain English | One-time or every time? |
|---|---|---|
| `./isaaclab.sh -i rsl_rl` | `-i` = **install**. Tells Lab's own installer to pull in the `rsl_rl` pip package (the PPO library itself, from Rudin et al.) into Lab's bundled Python environment. This is a Lab *dependency*, not your code. | **Once** per Isaac Lab install — re-run only if you reinstall/upgrade Lab, or delete/rebuild its env. |
| `./isaaclab.sh -p -m pip install -e /path/to/.../mars_rover_rl` | `-p` = **run the following using Lab's own Python interpreter** (not your Mac `.venv`, not system Python). `-m pip install -e ...` is the *same* editable install from Ch.2 §2.9 — but this time it registers `mars_rover` inside **Lab's** environment instead of your Mac venv, which is what lets `import mars_rover.envs` succeed when Isaac Sim's own Python process runs it. | **Once**, then automatically stays in sync — `-e` (editable) means future edits to your `.py` files apply without reinstalling. Only re-run if you move the repo path or add new top-level packages. |
| `./isaaclab.sh -p scripts/.../train.py --task Mars-Perseverance-Nav-v0 --headless --num_envs 64 --max_iterations 2000` | The actual training launch. Runs Lab's **own, pre-built** PPO training entry script (you didn't write this file — it ships with Isaac Lab) using Lab's Python. | **Every time** you start a training run. |

### The `train.py` flags, one at a time

| Flag | Meaning | Where else you've seen this number |
|---|---|---|
| `--task Mars-Perseverance-Nav-v0` | The Gym id you registered in §7.5. This is how the generic Lab script finds *your* env cfg and *your* agent cfg — it doesn't know anything about Perseverance by itself. | `gym.register(id=...)` in §7.5 |
| `--headless` | Skip opening the live 3D viewport window; physics/PPO still run at full speed, metrics still log normally. | §7.9.1 — this is the flag that trades away the "live viewport" option |
| `--num_envs 64` | How many parallel rover clones to simulate. Overrides `NavigationEnvCfg.scene`'s default (`num_envs=64` in §7.4) from the command line, so you can try `--num_envs 4` for a quick check without editing the config file. | `MarsNavSceneCfg(num_envs=64, ...)` in §7.4 |
| `--max_iterations 2000` | How many collect-rollout + PPO-update cycles to run before stopping. Overrides `PerseverancePPORunnerCfg.max_iterations` (also `2000` by default, in §7.6) from the command line. | `max_iterations = 2000` in §7.6 |

**Why CLI flags can override the same values the config classes already set:** Lab's `train.py` builds the config objects first, then applies any matching CLI args on top — handy for quick experiments (`--num_envs 4 --max_iterations 50` to sanity-check a run in a minute) without permanently editing `navigation_env_cfg.py` or `rsl_rl_ppo_cfg.py`.

### How the script finds your hyperparameters without a `--ppo_config` flag

Lab's script will look up `rsl_rl_cfg_entry_point` from your `gym.register` kwargs (§7.5) — that's the string `"mars_rover.envs.navigation.agents.rsl_rl_ppo_cfg:PerseverancePPORunnerCfg"`. So `--task Mars-Perseverance-Nav-v0` alone is enough for the script to locate *both* your env config *and* your PPO hyperparameters; the task id is effectively "one string that unlocks two config classes."

### Option B — Thin project `scripts/train.py` (educational)

Paste this only if you want a repo-local launcher; keep it aligned with your Lab version. Official Lab `train.py` evolves — when in doubt, prefer Option A.

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal RSL-RL training entry (educational).

Prefer Isaac Lab's scripts/reinforcement_learning/rsl_rl/train.py for production runs.
This file shows the conceptual spine: make env → wrap → OnPolicyRunner.learn.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import mars_rover.envs  # noqa: F401  # registers Gym id
from mars_rover.envs.navigation.agents.rsl_rl_ppo_cfg import PerseverancePPORunnerCfg


def main() -> None:
    agent_cfg = PerseverancePPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations

    env = gym.make("Mars-Perseverance-Nav-v0", num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(env)

    log_dir = os.path.join(
        "logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

---



## 7.9 Watching training — three different kinds of "seeing"

"Headless" understandably sounds like "you'll be flying blind." You won't — you just get **different views** depending on what question you're asking. There are three separate tools, and mixing them up is a common source of confusion:

```text
┌───────────────────────┬───────────────────────────┬─────────────────────────────┐
│ Tool                  │ Shows you                │ Speed cost                  │
├───────────────────────┼───────────────────────────┼─────────────────────────────┤
│ Live viewport         │ The rover(s) moving,      │ Slowest — GPU spends time   │
│ (non-headless)        │ right now, in real time   │ rendering every frame       │
├───────────────────────┼───────────────────────────┼─────────────────────────────┤
│ Video recording       │ A saved .mp4 of N steps,  │ Small — only renders during │
│ (--video flag)        │ watch it after the run    │ the recorded window         │
├───────────────────────┼───────────────────────────┼─────────────────────────────┤
│ TensorBoard           │ Numbers over time         │ ~Free — just reads logged   │
│ (metrics)             │ (reward, loss, entropy)   │ scalars, no rendering       │
└───────────────────────┴───────────────────────────┴─────────────────────────────┘
```

None of these are "the real training." They are three windows into the **same** underlying loop from §7.2. Pick the one that answers your current question.

### 7.9.1 Live viewport — "is the rover doing something sane at all?"

**What it is:** Isaac Sim's actual 3D window, updating every rendered frame, exactly like when you were clicking around in Chapter 3b.

**When to use it:** Very early on, or right after you fix a bug — a 30-second look often tells you more than 10 minutes of reward curves. Is it falling through the floor? Spinning in place? Driving off the edge instantly? You'll *see* it immediately.

**When not to use it:** For the actual multi-hour training run with 64+ parallel envs — rendering that many rovers every frame is wasted GPU work you'd rather spend on physics/learning.

**How:** simply **omit** `--headless` when launching:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Mars-Perseverance-Nav-v0 \
  --num_envs 4
```

Drop `num_envs` down (e.g. to 4) too — watching 64 overlapping rovers at once is visually useless. Isaac Lab tiles the parallel envs in a grid in the viewport by default, so a handful is much easier to actually read with your eyes.

### 7.9.2 Video recording — "let it run fast, but let me review a clip later"

**What it is:** Isaac Lab can render short video clips at intervals *while still training close to headless speed* the rest of the time, using Gymnasium's `RecordVideo` wrapper under the hood.

**When to use it:** Once early sanity checks pass and you're doing a real multi-hour run, but you still want to periodically eyeball behavior without babysitting a live window (or without a monitor at all — useful if the NVIDIA machine is headless-only / remote).

**How:**

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Mars-Perseverance-Nav-v0 \
  --headless \
  --video \
  --video_length 200 \
  --video_interval 2000
```

- `--video_length 200` → each clip is 200 simulation steps long
- `--video_interval 2000` → record a fresh clip every 2000 steps, so you get periodic snapshots of behavior as training progresses

Clips land under your log directory (`logs/rsl_rl/perseverance_nav/<timestamp>/videos/`) as `.mp4` files you can open normally. Watching clip #1 vs clip #20 side by side is one of the most satisfying ways to *see* learning happen — jerky random flailing early, smoother goal-seeking later.

**Requires:** `ffmpeg` installed on the NVIDIA machine (`sudo apt install ffmpeg` on Ubuntu).

### 7.9.3 TensorBoard — "is the *number* actually improving?"

This is the one already in the chapter — it doesn't show the rover at all, it shows the **grades** (Ch.5 rewards) and PPO internals over time. It's the tool you'll live in for most of a training run, precisely *because* it's cheap to check constantly without slowing anything down.

Training writes under something like:

```text
logs/rsl_rl/perseverance_nav/<timestamp>/
```

View with TensorBoard (from Lab's python):

```bash
./isaaclab.sh -p -m tensorboard.main --logdir /path/to/Mars_rover_rl/logs
```

Then open the printed `http://localhost:6006` URL in a browser.

### Curves that matter early


| Signal                 | Healthy early trend                                                         |
| ---------------------- | --------------------------------------------------------------------------- |
| Episode reward         | Noisy but slowly rising                                                     |
| Episode length         | May fall if success terminations increase, or rise if fewer instant crashes |
| Surrogate / value loss | Should not explode                                                          |
| Entropy                | Often decreases gradually as policy gets decisive                           |


If reward is flat forever: revisit Ch.5 weights and Ch.6 goal distance — not only LR.

### 7.9.4 Putting it together — a realistic workflow

```text
1) Fix a bug / first run of the day
       → non-headless, num_envs=4, watch live for ~1 minute
2) Confident it's not instantly broken
       → headless, --video --video_interval 2000, num_envs=64+, let it run for hours
3) Every so often while it runs
       → check TensorBoard (cheap, no slowdown, refresh anytime)
4) After training, want to show someone / evaluate properly
       → Ch.9's play.py: load a checkpoint, run a dedicated non-headless rollout
```

The "you still log metrics; you just don't render pretty windows" line from the glossary is really describing step 2 — TensorBoard logging is *always on* regardless of `--headless`, since it's just numbers being written to disk. Rendering (viewport or video) is the expensive part `--headless` turns off.

---



## 7.10 Realistic expectations (important honesty)

You can paste all Chapter 7 files today, but **training will fail** until:

1. `MarsNavSceneCfg.robot` is a real `ArticulationCfg` pointing at your USD
2. Wheel joint names match `ActionsCfg`
3. Contact sensor paths are valid
4. You run on NVIDIA with Isaac Sim / Lab + `rsl_rl` installed

Chapter 7’s job is to make the **learning pipeline** clear and ready so the moment the robot asset is wired, you are not inventing training infrastructure under panic.

---



## 7.11 Checklist

1. Explain gym registration vs env cfg vs agent cfg in your own words.
2. Paste `navigation_env_cfg.py`.
3. Paste gym.register into navigation `__init__.py` and ensure package import triggers it.
4. Paste `PerseverancePPORunnerCfg`.
5. Paste `zero_agent.py`.
6. On NVIDIA (when robot ready): zero-agent smoke test → then Lab `train.py`.
7. Write notes: first hyperparams you might change if reward is stuck.

---



## 7.12 Looking ahead

- **Ch.8** — cameras / reach-avoid ideas on top of a working policy loop  
- **Ch.9** — play checkpoints, metrics, failure analysis

---



## 7.13 North star

**PPO is not magic — it is a disciplined loop that improves a policy using the world and gradebook you built; Chapter 7 is where that loop finally runs.**