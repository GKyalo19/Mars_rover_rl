# Chapter 9 — Evaluation

> **How we work**
> - Explanations + full code live **in this chapter**.
> - **You** create files and paste while reading.
> - I will **not** write into your `source/` package unless you ask.
>
> **Goal:** Load a trained checkpoint (Ch.7) and answer two separate questions: *"does it work?"* (metrics) and *"how does it fail?"* (failure-mode analysis) — not just *"does it look cool in the viewport?"*
> **Depends on:** Ch.5 (reward/termination term names), Ch.7 (a checkpoint actually exists under `logs/rsl_rl/...`), Ch.8 (optional — the reach-avoid shield, if you built it).
> **Hard gate:** Same root cause as Ch.7/8 — you need at least one completed training run (robot no longer `MISSING`, `train.py` has produced a `model_*.pt`). Everything below can be read and pasted before that; it cannot be *run* before that.
> **Next:** none scheduled — Ch.9 closes the Phase A loop. Revisit Ch.5/6/7 hyperparameters based on what Ch.9 tells you, then come back here.

---

## 9.0 Bigger picture — training produces a checkpoint; evaluation produces a verdict

Chapter 7 ends with numbers going up on a TensorBoard curve. That is **necessary but not sufficient** — a rising `Episode_Reward/progress_to_goal` curve does not by itself tell you the rover reaches goals reliably, or *how* it fails when it doesn't.

```text
Ch.7: train.py  ──▶  checkpoint (model_1999.pt)
                            │
                            ▼
Ch.9: play.py / eval loop
        │
        ├──▶ "does it work?"   → success rate, collision rate, time-to-goal
        └──▶ "how does it fail?" → which termination fired, on which kind of episode
```

### Industry parallel

Shipping an ML model on reward/loss curves alone is a known anti-pattern. Robotics and self-driving teams hold out **evaluation scenarios** (Ch.6 §6.0 already named this layer) and track a small dashboard of task-specific metrics — success rate, safety-violation rate, efficiency — separately from the training loss. Chapter 9 builds that dashboard for Perseverance, at a scale appropriate for a Phase A project.

### Two evaluation modes, don't confuse them

| Mode | `num_envs` | Headless? | Purpose |
|---|---|---|---|
| **Watch** | small (2–4) | No | "Is it doing something sane, and does it *look* like real navigation?" — human judgment |
| **Measure** | larger (32–128) | Yes | "What's the success rate over many episodes?" — statistics, not vibes |

Both load the *same* checkpoint. They differ only in how many envs you run and whether you bother rendering — the exact same headless/viewport trade-off from Ch.7 §7.9 applies here too.

---

## 9.1 Jargon, explained properly

### Checkpoint loading / resume

Ch.7's `PerseverancePPORunnerCfg` already has the fields for this: `resume`, `load_run` (regex over run folder names), `load_checkpoint` (regex over `model_*.pt` filenames). Evaluation is the first place you actually use them — training doesn't need to load a checkpoint, evaluation always does.

### Deterministic vs. stochastic action

Ch.1 §? already flagged this distinction: **during training**, PPO **samples** from the Gaussian policy output (exploration is the point). **During evaluation**, you almost always want the **mean** of that Gaussian instead — the policy's actual best guess, not a noisy sample. RSL-RL's inference policy typically exposes this as a `deterministic` flag or simply returns the mean when you call the exported inference function rather than the training-time `act` method. Mixing this up gives you noisier, worse-looking eval rollouts than the policy actually produces.

### Rollout vs. episode (evaluation-specific nuance)

In Ch.7, "rollout" meant "a batch of steps collected before a PPO update." In evaluation, you don't update anything — you just care about **episodes**: each env's reset-to-termination span. One evaluation *run* contains many parallel envs, each producing a stream of episodes over time.

### Episode outcome / termination reason

Ch.5 gave you four termination terms: `goal_reached`, `illegal_contact` (collision), `time_out`, `farther_than_allowed` (out of bounds). Every finished episode ends because **exactly one** of these fired first. Evaluation's job is to count *which one*, across many episodes — that count **is** your metrics dashboard.

### Success rate / collision rate / timeout rate / out-of-bounds rate

The fraction of finished episodes whose termination reason was `goal_reached` / `illegal_contact` / `time_out` / `farther_than_allowed`, respectively. These four should sum to ~100% of finished episodes (modulo the rare edge case of two terms firing the same step).

### Time-to-goal

Episode length (steps or seconds), measured **only over successful episodes**. Averaging this across *all* episodes (including crashes/timeouts) would be meaningless — a crash at step 3 doesn't mean "fast," it means "failed fast."

### Path efficiency

\[
\text{efficiency} = \frac{\text{straight-line distance}(\text{start}, \text{goal})}{\text{actual distance traveled}}
\]

A value near 1.0 means the rover drove almost straight to the goal; well below 1.0 means it wandered, even if it eventually succeeded. This is the metric that catches "technically reached the goal, but took a bizarre route" — something success rate alone can't see.

### Failure mode / failure mode taxonomy

A **failure mode** is a *pattern* among failures, not just a count. "23% collision rate" is a metric; "most collisions happen in the first 2 seconds near spawn, when goals are sampled behind nearby rocks" is a **failure mode** — actionable, points back at a specific earlier chapter (here, Ch.6's goal-sampling ranges).

### Ablation

Re-running evaluation with one thing changed (a hyperparameter, an observation term, the shield on/off) to isolate *that one thing's* effect on the metrics dashboard. Cheap ablations you already have the pieces for: with/without `height_scan` (Ch.8 Track A), with/without the reach-avoid shield (Ch.8 Track C).

### Held-out evaluation seed

Using a **different** terrain/goal random seed for evaluation than whatever training used, so you're not just measuring "did it memorize this exact map." Ch.6's `MarsProceduralTerrainCfg.seed=42` was fixed for training debugging — evaluation is a good place to finally set `seed=None` or a different fixed value.

---

## 9.2 Option A — Isaac Lab's official RSL-RL play script (recommended for watching)

Mirrors Ch.7 §7.8's Option A pattern — Lab ships a play script alongside its train script:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Mars-Perseverance-Nav-v0 \
  --num_envs 4 \
  --load_run <run_folder_name> \
  --checkpoint model_1999.pt
```

This runs **non-headless by default** (you want to see it), uses the same `--task` lookup mechanism from Ch.7 §7.8, and defaults to **deterministic** action selection since that's what "evaluation" is supposed to mean. It's the fastest way to get a live look at a checkpoint. It does **not** give you a success-rate dashboard — for that, Option B below.

---

## 9.3 Option B — custom `scripts/play.py` with metrics (full code to copy)

### File: `scripts/play.py`

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Load a Perseverance-nav checkpoint and report success/collision/timeout
metrics over many episodes.

Prefer Isaac Lab's own play.py (Ch.9 §9.2) just to *watch* a checkpoint.
Use this one when you want the metrics dashboard (§9.4).

Example:
  ./isaaclab.sh -p scripts/play.py \
    --task Mars-Perseverance-Nav-v0 \
    --checkpoint logs/rsl_rl/perseverance_nav/2026-07-15_10-00-00/model_1999.pt \
    --num_envs 32 \
    --num_episodes 100 \
    --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a Perseverance nav checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=100, help="Stop once this many episodes finish")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import mars_rover.envs  # noqa: F401  # registers Gym id
from mars_rover.envs.navigation.agents.rsl_rl_ppo_cfg import PerseverancePPORunnerCfg


def main() -> None:
    agent_cfg = PerseverancePPORunnerCfg()

    env = gym.make("Mars-Perseverance-Nav-v0", num_envs=args_cli.num_envs)
    env = RslRlVecEnvWrapper(env)

    # log_dir=None: we are not writing new TensorBoard logs, only reading a
    # checkpoint into the same network architecture the runner expects.
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Per-env bookkeeping so we can attribute each finished episode to a
    # termination reason and a duration.
    num_envs = env.unwrapped.num_envs
    episode_steps = torch.zeros(num_envs, device=env.unwrapped.device)
    outcomes: dict[str, int] = {"goal_reached": 0, "illegal_contact": 0, "time_out": 0, "farther_than_allowed": 0}
    success_times: list[float] = []
    finished_episodes = 0

    obs, _ = env.get_observations()
    with torch.inference_mode():
        while finished_episodes < args_cli.num_episodes:
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            episode_steps += 1

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for env_id in done_ids.tolist():
                reason = _termination_reason(env.unwrapped, env_id)
                outcomes[reason] = outcomes.get(reason, 0) + 1
                if reason == "goal_reached":
                    success_times.append(episode_steps[env_id].item())
                episode_steps[env_id] = 0
                finished_episodes += 1

    _report(outcomes, success_times, finished_episodes)
    env.close()


def _termination_reason(env, env_id: int) -> str:
    """Which termination term fired for this env's just-finished episode.

    Isaac Lab's TerminationManager tracks each term's last-computed boolean
    tensor. ``get_term`` is the concept to look for; verify the exact method
    name against your installed Lab 3.x build (compare with
    ``isaaclab.managers.TerminationManager``) if this errors.
    """
    term_manager = env.termination_manager
    for name in ("goal_reached", "illegal_contact", "time_out", "farther_than_allowed"):
        term_tensor = term_manager.get_term(name)
        if bool(term_tensor[env_id]):
            return name
    return "unknown"  # should not happen if the four Ch.5 terms are exhaustive


def _report(outcomes: dict[str, int], success_times: list[float], total: int) -> None:
    print(f"\n[play] {total} episodes evaluated")
    for reason, count in outcomes.items():
        pct = 100.0 * count / max(total, 1)
        print(f"  {reason:22s} {count:5d}  ({pct:5.1f}%)")
    if success_times:
        avg_steps = sum(success_times) / len(success_times)
        print(f"  avg time-to-goal (successes only): {avg_steps:.1f} steps")
    else:
        print("  no successful episodes — see §9.5 failure-mode table before touching hyperparameters")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

### How to read this script

| Piece | Plain English |
|---|---|
| `runner.load(args_cli.checkpoint)` | Loads saved weights into the same network architecture `PerseverancePPORunnerCfg` describes — must match whatever cfg the checkpoint was trained with |
| `runner.get_inference_policy(...)` | Returns a callable that goes obs → deterministic action, without exploration noise (§9.1's deterministic-vs-stochastic point) |
| `episode_steps` | Per-env counter, reset to 0 whenever that env's episode ends — this is where "time-to-goal" comes from |
| `dones.nonzero(...)` | Which envs *just* finished this step — Isaac Lab resets those envs automatically next step, so we must capture the reason *now* |
| `_termination_reason` | Checks Ch.5's four termination terms in order and reports whichever one is `True` for that env |
| `--num_episodes` | Stop condition — run until you've *seen* this many completed episodes, not a fixed step count, since different policies finish episodes at different rates |

**Honesty note (same spirit as Ch.7/8):** `env.termination_manager.get_term(name)` is the *concept* — Isaac Lab manager classes expose each term's last-computed tensor, but the exact accessor name can differ slightly across Lab 3.x point releases. If it errors, check `isaaclab.managers.TerminationManager`'s actual API in your installed version; the logic of "read the boolean tensor for a specific named term" is what to preserve.

---

## 9.4 Reading the dashboard

Running §9.3 prints something like:

```text
[play] 100 episodes evaluated
  goal_reached           61  ( 61.0%)
  illegal_contact        22  ( 22.0%)
  time_out               14  ( 14.0%)
  farther_than_allowed     3  (  3.0%)
  avg time-to-goal (successes only): 187.4 steps
```

| Metric | Healthy Phase A ballpark | What it's telling you |
|---|---|---|
| Success rate (`goal_reached`) | Rising run-over-run; specific target depends on your task difficulty | The one number that matters most — everything else explains its shortfall |
| Collision rate (`illegal_contact`) | Should trend down as training progresses | High + flat → revisit Ch.5 collision weight or Ch.6 obstacle density |
| Timeout rate (`time_out`) | Should shrink as the policy gets more decisive | High → policy dithers; check entropy/progress-reward weighting (Ch.7 §7.6 table) |
| Out-of-bounds rate (`farther_than_allowed`) | Should be small | High → policy is actively driving *away* from goals; check sign conventions in `heading_to_goal` / `progress_to_goal` |
| Path efficiency | Add this yourself (see below) once success rate is non-trivial | Distinguishes "reaches the goal cleanly" from "eventually stumbles into it" |

### Adding path efficiency (extension, not pasted above)

The script above doesn't track traveled distance yet — that needs summing per-step displacement per env between reset and termination. If you want it: accumulate `torch.norm(pos_t - pos_{t-1})` into a per-env running total alongside `episode_steps`, snapshot it at episode end next to `success_times`, and divide by the goal command's straight-line distance at reset. Left as a follow-up once the four termination-reason counts above are working — don't add complexity you don't need yet.

---

## 9.5 Failure-mode analysis — a debugging map back to earlier chapters

This is the part a raw percentage table can't do by itself: **connecting an outcome back to a *cause* in an earlier chapter.**

| Dominant failure | Likely root cause | Chapter to revisit |
|---|---|---|
| Mostly `illegal_contact`, early in episodes (low `episode_steps` at death) | Goals or spawns land too close to obstacles; reset pose overlaps a rock | Ch.6 §6.6 (`CommandsCfg` ranges), Ch.6 §6.7 (`reset_base` pose ranges) |
| Mostly `illegal_contact`, late in episodes | Policy learned to approach goals but not to avoid rocks *near* them | Ch.5 collision weight too small relative to progress weight |
| Mostly `time_out` | Policy is slow/indecisive, or episode length budget is too tight for the goal distances you sampled | Ch.6 §6.6 `pos_x`/`pos_y` ranges (goals too far), or Ch.7 `episode_length_s` |
| Mostly `farther_than_allowed` | Heading/progress reward sign or scale error — policy is optimizing the wrong direction | Ch.4 `heading_to_goal` sign convention, Ch.5 `progress_to_goal` implementation |
| High success rate but low path efficiency | Reward taught "eventually get there," not "get there directly" — common when progress reward is too sparse or entropy coefficient too high | Ch.5 progress reward density, Ch.7 `entropy_coef` |
| Collisions cluster around `--num_envs` cliques of similar goals | Not a policy bug — a Ch.6 terrain/goal-sampling artifact (e.g. a systematically bad region of the map) | Ch.6 §6.4 obstacle placement, §6.6 goal ranges |

**The discipline this table encodes:** before touching PPO hyperparameters (Ch.7), first ask whether the failure pattern points at rewards (Ch.5) or world generation (Ch.6) instead — those are usually cheaper and more diagnostic fixes than a hyperparameter sweep.

---

## 9.6 Optional: shielded vs. unshielded comparison (hooks into Ch.8 Track C)

If you built Ch.8's `reach_avoid_shield.py`, run §9.3's evaluation loop twice — once passing `actions` straight to `env.step`, once passing `shield_action(actions, obs["height_scan"])` (see Ch.8 §8.3 for how `height_scan` reaches the obs dict) — and diff the two dashboards:

| Metric | Unshielded | Shielded | Interpretation |
|---|---|---|---|
| Collision rate | higher | lower (hopefully) | Shield is catching cases the policy itself hasn't learned to avoid |
| Success rate | baseline | may drop slightly | Overly conservative `clearance_threshold` can stop the rover short of goals it would've threaded past safely |
| Time-to-goal | baseline | may rise slightly | Stopping-and-recomputing costs a few steps per intervention |

If shielded success rate drops noticeably, that's a signal `clearance_threshold` (Ch.8 §8.4) is too conservative for this map — tune it down before concluding the shield "doesn't work."

---

## 9.7 Ablations (optional, do only once success rate is non-trivial)

Cheap ablations available from pieces you already have:

1. **With / without `height_scan`** (Ch.8 Track A) — does local terrain awareness actually move the needle, or was goal-relative distance/heading alone already carrying the policy?
2. **Reward weight sweep** (Ch.5 §5.x `RewardsCfg` weights) — halve/double the collision weight, re-train, re-evaluate; compare dashboards, not just training curves.
3. **Goal-distance curriculum** (Ch.6 §6.6 tuning tip) — evaluate a policy trained on close goals against a held-out set of *far* goals, to see how well it generalizes past its training distribution.

Each ablation is: change one config value → retrain (Ch.7) → re-run §9.3 → compare dashboards side by side. Resist changing two things at once; you'll lose the ability to attribute the effect.

---

## 9.8 Files map (this chapter)

```text
scripts/
└── play.py     # §9.3 — metrics-reporting evaluation loop (new)
```

Everything else in Chapter 9 is a way of *reading* the numbers this one script produces, plus the failure-mode table (§9.5) that turns those numbers into next actions in earlier chapters.

---

## 9.9 Realistic expectations

1. You need a real checkpoint — `--checkpoint` pointing at an actual `model_*.pt` file under `logs/rsl_rl/perseverance_nav/<timestamp>/`. Nothing here runs without one.
2. The four termination reasons in `_termination_reason` are exactly Ch.5's `TerminationsCfg` terms by name — if you renamed any of them, update this list to match, or you'll silently fall into the `"unknown"` bucket.
3. Low success rate on your **first** checkpoint is normal, not a bug — it's exactly what §9.5's failure-mode table is for.
4. Path efficiency and shielded/unshielded comparisons (§9.4, §9.6) are optional extensions layered on top of the core four-way outcome count — get that working first.

---

## 9.10 Checklist

1. Run Option A (§9.2) once, non-headless, `num_envs=4` — does the rover's behavior *look* plausible?
2. Paste `scripts/play.py` (§9.3); run it headless with `num_envs=32`, `num_episodes=100` against your latest checkpoint.
3. Read the four-way outcome table (§9.4) and identify the dominant failure mode.
4. Use §9.5's table to point that failure mode at a specific earlier chapter/config value.
5. Make **one** change, retrain, re-evaluate, compare dashboards (§9.7's discipline).
6. (Optional) If Ch.8 Track C exists, run the shielded-vs-unshielded comparison (§9.6).
7. Write one paragraph: current success rate, dominant failure mode, and the single next experiment you'd run.

---

## 9.11 North star

**A reward curve going up answers "is PPO working?" A success-rate dashboard with a failure-mode table answers "is the rover working?" — Chapter 9 is where Phase A stops being an infrastructure project and starts being an evaluated navigation policy.**
