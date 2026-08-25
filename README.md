# Mars_rover_rl

Train the **2020 Mars Perseverance Rover** to navigate Mars-like terrain with reinforcement learning.

| | |
|--|--|
| **Sim** | Isaac Sim **6.0.1** |
| **RL framework** | Isaac Lab **v3.0.0-beta2.patch1** |
| **Python (ASUS)** | **3.12** (Lab 3.0 Beta 2) |
| **Algorithm** | PPO via RSL-RL |
| **Robot** | Perseverance (six wheel velocity targets) |
| **Dev** | Author on Mac → train on NVIDIA |
| **Baseline v1** | `action_dim = 6`, `obs_dim = 19`, flat terrain default |

If the ASUS install uses a different Lab tag, record that tag in the run log rather than silently drifting.

**Do not start PPO until Gates A–D pass** (`zero_agent.py` first).

## Baseline v1 contract

Default Gym task: `Mars-Perseverance-Nav-v0` (flat). Rough rocks: `Mars-Perseverance-Nav-Rough-v0`.

**Action** `[FL, ML, RL, FR, MR, RR]` maps to USD joints:

```text
action[0] → Joint_Wheel_FL
action[1] → Joint_Wheel_ML
action[2] → Joint_Wheel_RL
action[3] → Joint_Wheel_FR
action[4] → Joint_Wheel_MR
action[5] → Joint_Wheel_RR
```

Chassis rigid body: **`Body`**. Articulation root under the spawned asset: **`Rover`**.

Isaac Lab 3.0 quaternions are **`(x, y, z, w)`**. Identity is `(0, 0, 0, 1)`, not the Lab 2 `(w, x, y, z)` identity.

- **Observations:** proprioception only — goal `(dx, dy)` in the rover **local horizontal** frame, `vx, vy`, `wz`, six wheel speeds, roll/pitch, **previous** action.
- **Goal:** one `command_manager` term (`target_pose`). Observations, rewards, and terminations all read that same state.
- **Rewards (first PPO run):** temporal progress, goal, body collision, unsafe attitude, idle-after-duration, time. Extra terms stay unimplemented in the default cfg.
- **Vision / height-scan / reach-avoid / curriculum / domain randomization:** in the repo, **disabled** from the default env.

## Rover USD (not in git)

Keep the Isaac Sim authoring scene (`Mars Rover Suspension Fixed.usd` or similar) as the master. For Isaac Lab, export a **cleaned** `Perseverance_Rover.usd`:

- Keep `Rover` and `_materials`
- **Remove** `GroundPlane`, `env_light`, and `PhysicsScene`

Place it at:

`source/mars_rover_rl/mars_rover/assets/robots/perseverance/Perseverance_Rover.usd`

or set `MARS_ROVER_USD`. Drive folder: [Google Drive assets](https://drive.google.com/drive/u/0/folders/1oYXenyja_fFVZ6yVrJQeySnr2uDTPiPS).

Joint/body names live in `mars_rover/mdp/kinematics.py` (`ROVER_WHEEL_JOINTS`, `CHASSIS_BODY_NAME`).

## Install

NVIDIA / Isaac Lab Python 3.12:

```text
git clone <this-repo>
cd Mars_rover_rl
./isaaclab.sh -p -m pip install -e source/mars_rover_rl
```

Mac (no Isaac): `.venv/bin/python -m pytest tests`

## Scripts (NVIDIA)

```text
./isaaclab.sh -p scripts/zero_agent.py --num_envs 1
./isaaclab.sh -p scripts/train.py --task Mars-Perseverance-Nav-v0 --num_envs 4 --max_iterations 5
./isaaclab.sh -p scripts/play.py --checkpoint logs/rsl_rl/perseverance_nav/<run>/model_*.pt
./isaaclab.sh -p scripts/evaluate.py --checkpoint <path> --num_episodes 100 --out metrics.json
```

`evaluate.py` reports success rate, distance, time-to-goal, collisions, episode length — **not** reward as the science metric.

## Gates (Baseline v1 is complete only after all pass on the ASUS)

| Gate | Check |
|------|--------|
| **A Asset** | Cleaned USD, `Rover` root, `Body`, six `Joint_Wheel_*` in order, 6-D / 19-D |
| **B Physics** | +ω all wheels → forward (measure axis signs; do not guess); no teleport |
| **C Steering** | Left/right differential → turn; zero → stop |
| **D Env** | Reset/step tensors; `zero_agent.py` without NaNs |
| **E PPO** | Tiny `train.py` then play/evaluate |

## Docs

Chapter walkthrough: [docs/README.md](docs/README.md). Earlier chapters use conceptual names (`base_link`, `wheel_FL`); the **USD names above are authoritative**.

## License / credit

Architecture patterns inspired by [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab) (Mortensen & Bøgh). This repo is a separate Perseverance-focused project.
