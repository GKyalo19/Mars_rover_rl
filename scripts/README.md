# Scripts

Run on the NVIDIA machine with Isaac Lab's Python (`./isaaclab.sh -p …`) after `pip install -e source/mars_rover_rl` and the Perseverance USD.

| Script | Purpose |
|--------|---------|
| `zero_agent.py` | Reset/step with zero 6-D actions; asserts obs_dim=19 |
| `train.py` | RSL-RL PPO (smoke defaults: 4 envs, 5 iterations) |
| `play.py` | Load a checkpoint and run the policy |
| `evaluate.py` | Success / distance / time / collisions over `--num_episodes` (not reward) |
| `rl_common.py` | Shared Gym cfg loader for the scripts above |
| `inspect_blend.py` | Blender-only asset inspect (Mac-safe) |

Example:

```text
./isaaclab.sh -p scripts/train.py --task Mars-Perseverance-Nav-v0 --num_envs 4 --max_iterations 5 --seed 42
```

Rough terrain: `--task Mars-Perseverance-Nav-Rough-v0`.
