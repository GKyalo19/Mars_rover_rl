# Mars_rover_rl documentation

Living notes for training Perseverance navigation with **Isaac Sim 6.0.1**, **Isaac Lab v3.0.0-beta2.patch1**, and **RSL-RL (PPO)**.

**Default MDP is Baseline v1:** 6-D wheel velocities, 19-D proprioception, flat terrain (`Mars-Perseverance-Nav-v0`). Chapter 8 vision / reach-avoid stays optional and is not the default policy.

**USD naming (authoritative):** chassis `Body`, joints `Joint_Wheel_FL` … `Joint_Wheel_RR`, articulation child `Rover`. Earlier chapters use conceptual names such as `base_link` and `wheel_FL` — those are historical. The final USD names are what Isaac Lab must match.

**Quaternions (Isaac Lab 3.0):** all orientation data is `(x, y, z, w)`, not `(w, x, y, z)`. Identity is `(0, 0, 0, 1)`.

Docs and code advance together: each chapter lands before (or with) the modules it describes. If a chapter still shows `[v, ω]` as the default action, treat that as historical; the live env uses six wheel joints.

## Chapters

| # | Chapter | Status |
|---|---------|--------|
| 1 | [Software architecture & project blueprint](chapters/01_software_architecture.md) | Done |
| 2 | [Install, tooling & starting the codebase](chapters/02_install_and_tooling.md) | Done — packaging + bigger picture |
| 3 | [Rover asset pipeline (Blend → USD)](chapters/03_rover_asset_pipeline.md) | Blender guide + concepts |
| 3b | [Isaac articulation & physics](chapters/03b_isaac_articulation_and_physics.md) | Beginner Isaac Sim/Lab guide |
| 4 | [Observation & action spaces](chapters/04_observation_and_action_spaces.md) | Done |
| 5 | [Rewards & terminations](chapters/05_rewards_and_terminations.md) | Done — copy code from the chapter |
| 6 | [Terrain, goals & obstacles](chapters/06_terrain_goals_obstacles.md) | Done — copy code from the chapter |
| 7 | [PPO training loop (RSL-RL)](chapters/07_ppo_training_loop.md) | Done |
| 8 | [Vision & reach-avoid](chapters/08_vision_and_reach_avoid.md) | Done — optional tracks A/B/C |
| 9 | [Evaluation](chapters/09_evaluation.md) | **You are here** — metrics + failure modes |

## Also

- [Glossary](glossary/rl_jargon.md) — short definitions as we introduce terms
- Inspiration (not a fork): [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab)
