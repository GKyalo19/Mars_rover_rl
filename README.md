# Mars_rover_rl

Train the **2020 Mars Perseverance Rover** to navigate Mars-like terrain with reinforcement learning.

| | |
|--|--|
| **Sim** | Isaac Sim 6.0 |
| **RL framework** | Isaac Lab 3.0 |
| **Algorithm** | PPO via RSL-RL |
| **Robot** | Perseverance (single-rover focus) |
| **Dev** | Author on Mac → train on NVIDIA |

## Documentation first

We are building this in chapters. Start here:

**[Chapter 1 — Software architecture & PPO primer](docs/chapters/01_software_architecture.md)**

Full doc map: [docs/README.md](docs/README.md)

## Repo status

Scaffold + Chapter 1 only. Package folders exist under `source/mars_rover_rl/`; training code comes in later chapters.

## Layout (preview)

```text
source/mars_rover_rl/   # Env, assets, MDP terms
configs/                # RSL-RL + env presets
scripts/                # train / play (later)
docs/chapters/          # Design + math walkthrough
```

## License / credit

Architecture patterns inspired by [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab) (Mortensen & Bøgh). This repo is a separate Perseverance-focused project.
