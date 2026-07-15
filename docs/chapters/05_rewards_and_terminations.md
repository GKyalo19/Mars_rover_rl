# Chapter 5 — Rewards & Terminations

> **How we work from now on**  
> - I put **explanations + full code in this chapter**.  
> - **You** create the files and paste the code in while reading.  
> - I will **not** write into `source/mars_rover/...` for you unless you explicitly ask.  
>
> **Goal of this chapter:** Teach the math and Isaac Lab wiring for *what counts as good driving* (rewards) and *when an episode ends* (terminations).  
> **Depends on:** Chapters 1–4 (MDP idea, packaging, physics intuition, observation/action spaces).  
> **Does not finish yet:** Mars terrain generator (Ch.6), PPO `train.py` (Ch.7).

---

## 5.0 Bigger picture — why this chapter exists

So far the puzzle looks like this:

| Piece | Status | Role |
|-------|--------|------|
| Package `mars_rover` | Done (Ch.2) | Code can be imported |
| Physical rover ideas | In progress (Ch.3 / 3b) | Wheels, joints, drives |
| Observations \(o_t\) | Designed (Ch.4) | What the policy **sees** |
| Actions \(a_t = [v,\omega]\) | Designed (Ch.4) | What the policy **commands** |
| **Rewards \(r_t\)** | **This chapter** | How we **grade** each step |
| **Terminations** | **This chapter** | When we **stop** the episode |
| Terrain + goals | Ch.6 | The exam questions |
| PPO training | Ch.7 | How grades update the policy |

### The one idea to keep

Reinforcement learning does **not** learn from you saying “turn left here.”  
It learns from a **number** returned every step: the **reward**.

If the reward says “getting closer to the goal is good” and “hitting a rock is very bad,” then PPO’s job (Ch.7) is to change the policy so that future actions tend to collect more of those good numbers over time.

**Industry parallel:**  
Think of the reward function as a **product specification written in math**. In self-driving and robot navigation teams, a huge amount of engineering time goes into reward design (sometimes called **reward shaping**), because a wrong spec teaches the wrong skill — even if the neural net and simulator are perfect.

**Path planning connection (Phase A):**  
We still do not run A\* as a separate module. The “plan” emerges because:

- progress toward the goal is rewarded  
- collisions / leaving the map end the episode with penalties  
- jerky steering is discouraged  

A sequence of actions that maximize expected return *is* a path, discovered by trial and error across thousands of parallel simulated drives.

---

## 5.1 Jargon, explained properly (not cryptic)

### Reward \(r_t\)

A **reward** is a scalar number the environment gives the agent after it takes an action at time \(t\).

- Positive values usually mean “that was desirable.”  
- Negative values usually mean “that was undesirable.”  
- Zero means “nothing special happened this step.”

Important subtlety: the agent does **not** need to understand English. It only sees numbers. So if you want “don’t reverse unnecessarily,” you must express that as a numeric penalty, not as a comment in the code.

### Return \(R_t\) (related, but not the same)

The **return** is the *discounted sum of future rewards* from time \(t\) onward:

\[
R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\]

- \(r_t\) = grade for **this** step  
- \(R_t\) = “how good was life from here on, total?”  

PPO (Ch.7) cares about improving expected return. Your job in Ch.5 is to make each \(r_t\) a meaningful grade so that high return corresponds to “safe arrival at the goal.”

### Discount factor \(\gamma\) (gamma)

\(\gamma\) is a number between 0 and 1 (often `0.99` in robotics RL).

- If \(\gamma\) is close to 1, the agent cares about long-term outcomes (reaching a distant goal).  
- If \(\gamma\) is small, the agent becomes short-sighted (only immediate rewards matter).

You will set \(\gamma\) in the PPO config later. You do **not** put \(\gamma\) inside each reward term. Reward terms only produce \(r_t\); the trainer discounts them when computing returns/advantages.

### Reward shaping

**Reward shaping** means designing intermediate rewards (not only a big bonus at the end) so learning is faster and more stable.

Example:

- Sparse-only design: `+100` only when the goal is reached, `0` otherwise.  
  The agent may wander forever before stumbling on success.  
- Shaped design: small positive reward every time distance-to-goal decreases, plus a large bonus on success.

Shaping is powerful and dangerous: if you over-shape, the agent may exploit loopholes (jargon: **reward hacking**) — e.g. circle near the goal forever collecting progress crumbs without finishing, unless you also reward success and terminate properly.

### Dense vs sparse rewards

| Type | Meaning | Example |
|------|---------|---------|
| **Sparse** | Feedback only on rare events | `+10` on goal, `-10` on crash |
| **Dense** | Feedback almost every step | progress each step, tiny time penalty |

Practical Mars navigation almost always uses a **mix**: dense progress + sparse success/collision.

### Weighted multi-term reward

Instead of one giant formula typed as a single expression, Isaac Lab (and industry codebases) split reward into **terms**, each with a **weight**:

\[
r_t = \sum_i w_i \, r^{(i)}_t
\]

Example:

- \(r^{\text{progress}}\) weight `+1.0`  
- \(r^{\text{collision}}\) weight `-5.0` (because the term itself might be `0` or `1`)  

This matches how RLRoverLab structures `rewards.py`: many small functions, weights in the config.

### Termination (done flag)

A **termination** answers: “Should this episode stop **now**?”

When an episode terminates, the environment resets for the next attempt (new goal / new spawn, once Ch.6 exists).

Common reasons:

| Termination | Meaning for Perseverance |
|-------------|--------------------------|
| **Success** | Close enough to the goal |
| **Collision** | Hit an obstacle hard enough |
| **Timeout** | Used too many steps / too many seconds |
| **Out of bounds** | Left the allowed map |
| **Stuck** (later) | No progress for a long time |
| **Flipped** (later) | Rolled over (advanced) |

### Truncation vs termination (Gymnasium language)

Modern Gymnasium distinguishes:

- **Terminated:** the task ended for a *semantic* reason (success, crash).  
- **Truncated:** the episode was *cut short* by a time limit even though the task wasn’t truly “over.”

Why care? Advantage estimators and bootstrapping treat “timeout at a good state” differently from “crashed.” Isaac Lab exposes this via termination terms with a `time_out=True` flag for the time-limit case.

### Reward term / termination term (Isaac Lab)

In the **manager-based** style:

- A **reward term** is a function `(env, **params) -> Tensor[num_envs]` producing per-env rewards for that ingredient.  
- A **termination term** is a function `(env, **params) -> Tensor[num_envs]` of booleans (or 0/1) saying which envs should end.

Configs attach **weights** (rewards) or **time_out** flags (terminations).

This is the same plugin idea as observation/action terms from Chapter 4 — another puzzle piece in the same board.

---

## 5.2 What “good driving” means for our rover (design spec)

Translate English goals into measurable signals:

| Human goal | Numeric idea | Term name (suggested) |
|------------|--------------|------------------------|
| Reach the destination | Large bonus when distance & heading are small enough | `reached_goal` |
| Keep moving toward the goal | Reward based on distance (closer = better) | `progress_to_goal` |
| Don’t hit rocks | Penalty when contact forces exceed a threshold | `collision` |
| Don’t thrash the joystick | Penalty when consecutive actions differ a lot | `oscillation` |
| Prefer forward driving | Small penalty when commanded \(v < 0\) | `reverse` |
| Don’t wander forever | Tiny negative reward every step | `time_penalty` |
| Don’t leave the workspace | End episode + penalty if too far / out of map | `out_of_bounds` (term + termination) |

### Phase A formula (conceptual)

\[
\begin{aligned}
r_t =\ & w_{\text{goal}}\, r^{\text{goal}}
      + w_{\text{prog}}\, r^{\text{progress}} \\
      &- w_{\text{col}}\, r^{\text{collision}}
      - w_{\text{osc}}\, r^{\text{oscillation}}
      - w_{\text{rev}}\, r^{\text{reverse}}
      - w_{\text{time}}\, r^{\text{time}}
\end{aligned}
\]

Exact weights are **starting guesses**. You will tune them after the first training runs (Ch.7–9). That tuning loop is normal in industry.

### Inspiration from RLRoverLab (ideas, not copy-paste numbers)

Their navigation rewards include:

- distance shaping that grows as you approach the target  
- a success bonus that can scale with remaining time (finish faster → better)  
- oscillation penalty between consecutive actions  
- reverse / heading soft constraints  
- collision from contact sensors  

We adopt the **structure**, then choose Perseverance-appropriate weights and our skid-steer action indices.

---

## 5.3 Files you will create (you type / paste)

Create these paths under your package (names assume package import `mars_rover`):

```text
source/mars_rover_rl/mars_rover/mdp/rewards.py
source/mars_rover_rl/mars_rover/mdp/terminations.py
source/mars_rover_rl/mars_rover/envs/navigation/config/rewards_cfg.py
source/mars_rover_rl/mars_rover/envs/navigation/config/terminations_cfg.py
```

Also extend (manually) your navigation `mdp/__init__.py` re-exports once those modules exist — only on the NVIDIA side if imports need Isaac Lab.

**Mac note:** Like Ch.4 observation stubs, reward functions that touch `env.command_manager` / contact sensors need Isaac Lab at runtime. You can still paste and read everything on the Mac; executing inside a live env waits for Ch.6–7 on NVIDIA.

---

## 5.4 Reward term functions — full code to copy

### File: `mars_rover/mdp/rewards.py`

Create the file, then paste:

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for Perseverance navigation (Phase A).

Each public function is an Isaac Lab *reward term*:

    fn(env, **params) -> Tensor[num_envs]

The RewardManager multiplies each term by a weight from RewardsCfg and sums.

Requires Isaac Lab at runtime (NVIDIA machine for real env stepping).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Dense shaping: larger when closer to the goal (XY).

    Why this form?
        A simple and stable choice (used in many rover/nav examples) is:

            1 / (1 + k * distance^2)

        - When distance is huge, reward ~ 0 (still room to improve).
        - When distance approaches 0, reward approaches 1.
        - Squaring distance makes far-away differences matter less than near-goal fine motion.

    We also divide by max episode length so summing over a full episode stays scaled
    reasonably when weights are O(1). This is a design choice, not a law of nature.
    """
    # Chapter 6 wires the command manager. Until then, if you unit-test without commands,
    # guard or stub. In a full env this block is the real implementation:
    target = env.command_manager.get_command(command_name)
    target_xy = target[:, :2]
    distance = torch.norm(target_xy, p=2, dim=-1)
    shaped = 1.0 / (1.0 + 0.11 * distance * distance)
    return shaped / env.max_episode_length


def reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
    distance_threshold: float = 0.25,
    heading_threshold: float = 0.2,
) -> torch.Tensor:
    """Sparse success bonus when near the goal and roughly aligned.

    Why heading too?
        Arriving sideways may be OK for some missions, but for a driving policy it often
        helps to require facing the target. Thresholds are meters / radians — tune later.

    Why scale by remaining time?
        Finishing earlier is better. If two policies both reach the goal, the faster one
        gets a larger bonus. This gently teaches efficiency without a huge hand-tuned
        speed reward.
    """
    target = env.command_manager.get_command(command_name)
    target_xy = target[:, :2]
    # Many Lab command layouts put heading error in a later column; adjust index when
    # you finalize CommandsCfg in Chapter 6. Placeholder assumes index 3 if present.
    heading_err = target[:, 3] if target.shape[-1] > 3 else torch.zeros(env.num_envs, device=env.device)

    distance = torch.norm(target_xy, p=2, dim=-1)
    time_left_frac = (env.max_episode_length - env.episode_length_buf) / env.max_episode_length
    success = (distance < distance_threshold) & (torch.abs(heading_err) < heading_threshold)
    return torch.where(success, 2.0 * time_left_frac, torch.zeros_like(distance))


def oscillation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize jerky changes between consecutive actions [v, ω].

    Intuition:
        If the policy flips from hard-left to hard-right every step, the rover twitches.
        Real Perseverance (and any hardware) prefers smoother commands.

    Implementation idea:
        Compare current action to previous action. If the delta is large, add a penalty.
    """
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    # Assuming action layout [linear_v, angular_w]
    lin_delta = action[:, 0] - prev_action[:, 0]
    ang_delta = action[:, 1] - prev_action[:, 1]
    lin_pen = torch.square(lin_delta)
    ang_pen = torch.square(ang_delta)
    return (lin_pen + ang_pen) / env.max_episode_length


def reverse_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Small penalty when commanded linear velocity is negative.

    Why?
        Reverse is sometimes necessary, but beginners' policies love reversing as a
        weird local optimum. A soft penalty says 'prefer forward unless you must reverse.'
    """
    v = env.action_manager.action[:, 0]
    return torch.where(v < 0.0, torch.ones_like(v) / env.max_episode_length, torch.zeros_like(v))


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty when contact sensor reports significant force against obstacles.

    Intuition:
        Visual meshes can interpenetrate slightly in sim; we use measured contact force
        as a practical collision signal (same spirit as RLRoverLab).

    Returns:
        1.0 for envs in collision, 0.0 otherwise (weight in cfg scales severity).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Force matrix layout can vary by sensor config; we take a robust aggregate.
    force_matrix = contact_sensor.data.force_matrix_w
    forces = torch.linalg.norm(force_matrix, dim=-1)
    # Sum across contact points / bodies depending on shape; flatten safely:
    forces_flat = forces.view(env.num_envs, -1).sum(dim=-1)
    in_collision = forces_flat > threshold
    return torch.where(in_collision, torch.ones(env.num_envs, device=env.device), torch.zeros(env.num_envs, device=env.device))


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant per-step cost to discourage endless wandering.

    Even a tiny penalty each step makes 'arrive sooner' better than 'arrive eventually,'
    when combined with a success bonus.
    """
    return torch.ones(env.num_envs, device=env.device) / env.max_episode_length
```

### How to read that file as puzzle pieces

| Function | Teaches the policy… | Dense/sparse |
|----------|---------------------|--------------|
| `progress_to_goal` | “Creep closer” | Dense |
| `reached_goal` | “Finishing matters a lot” | Sparse |
| `collision_penalty` | “Rocks hurt” | Sparse-ish event |
| `oscillation_penalty` | “Smooth steering” | Dense |
| `reverse_penalty` | “Forward preferred” | Dense when reversing |
| `time_penalty` | “Don’t stall forever” | Dense |

---

## 5.5 Termination term functions — full code to copy

### File: `mars_rover/mdp/terminations.py`

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for Perseverance navigation (Phase A).

Each function returns a boolean tensor [num_envs] — True means 'end this env's episode.'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End when the episode length budget is consumed.

    In Isaac Lab configs this is usually marked time_out=True so trainers treat it
    as truncation (time limit) rather than a semantic failure like a crash.
    """
    return env.episode_length_buf >= env.max_episode_length


def goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
    distance_threshold: float = 0.25,
) -> torch.Tensor:
    """End successfully when within distance_threshold of the goal XY."""
    target = env.command_manager.get_command(command_name)
    distance = torch.norm(target[:, :2], p=2, dim=-1)
    return distance < distance_threshold


def illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """End when collision contact force exceeds threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = contact_sensor.data.force_matrix_w
    forces = torch.linalg.norm(force_matrix, dim=-1).view(env.num_envs, -1).sum(dim=-1)
    return forces > threshold


def farther_than_allowed(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pose",
    max_distance: float = 25.0,
) -> torch.Tensor:
    """End if the rover is absurdly far from the goal (likely stuck/lost).

    This is a simple stand-in until Chapter 6 defines real map bounds.
    """
    target = env.command_manager.get_command(command_name)
    distance = torch.norm(target[:, :2], p=2, dim=-1)
    return distance > max_distance
```

### Termination map

| Function | Episode outcome story |
|----------|----------------------|
| `goal_reached` | “We did the mission.” |
| `illegal_contact` | “We crashed.” |
| `time_out` | “Time’s up — stop and reset.” |
| `farther_than_allowed` | “You’re lost — stop wasting sim time.” |

---

## 5.6 Config wiring — full code to copy

These configs do not compute rewards themselves. They are the **menu** that tells Isaac Lab which functions to call and how hard each signal should count.

### File: `envs/navigation/config/rewards_cfg.py`

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward weights / term wiring for Perseverance navigation."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from mars_rover.mdp import rewards as rewards


@configclass
class RewardsCfg:
    """Starting weights — expect to tune after first training curves.

    How to think about weights:
      - If collisions still happen constantly, make collision more negative.
      - If the rover never moves, increase progress / decrease time penalty cautiously.
      - If the rover twitches, increase oscillation penalty.
    """

    progress_to_goal = RewTerm(
        func=rewards.progress_to_goal,
        weight=5.0,
        params={"command_name": "target_pose"},
    )
    reached_goal = RewTerm(
        func=rewards.reached_goal,
        weight=10.0,
        params={
            "command_name": "target_pose",
            "distance_threshold": 0.25,
            "heading_threshold": 0.2,
        },
    )
    collision = RewTerm(
        func=rewards.collision_penalty,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor"),
            "threshold": 1.0,
        },
    )
    oscillation = RewTerm(
        func=rewards.oscillation_penalty,
        weight=-0.05,
    )
    reverse = RewTerm(
        func=rewards.reverse_penalty,
        weight=-0.5,
    )
    time_penalty = RewTerm(
        func=rewards.time_penalty,
        weight=-0.1,
    )
```

### File: `envs/navigation/config/terminations_cfg.py`

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Episode end conditions for Perseverance navigation."""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from mars_rover.mdp import terminations as terminations


@configclass
class TerminationsCfg:
    """When any term returns True for an env, that env resets."""

    time_out = DoneTerm(func=terminations.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=terminations.goal_reached,
        params={"command_name": "target_pose", "distance_threshold": 0.25},
    )
    collision = DoneTerm(
        func=terminations.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0},
    )
    out_of_bounds = DoneTerm(
        func=terminations.farther_than_allowed,
        params={"command_name": "target_pose", "max_distance": 25.0},
    )
```

---

## 5.7 How rewards + terminations connect to observations & actions

Walk one timestep in plain language:

1. Policy sees \(o_t\) (distance, heading, last action, … from Ch.4).  
2. Policy outputs \(a_t = [v,\omega]\).  
3. Action term turns that into wheel velocity targets; PhysX steps.  
4. Reward terms read the new situation (distance change, contacts, action jerks) and return numbers.  
5. Termination terms decide if this attempt is over.  
6. PPO stores \((o_t, a_t, r_t, o_{t+1}, done)\) for learning (Ch.7).

If rewards ignore collisions, a policy can learn to cut corners through rocks.  
If observations omit distance-to-goal, progress rewards cannot be predicted well.  
**Spaces (Ch.4) and rewards (Ch.5) must describe the same task.**

---

## 5.8 Common failure modes (reward hacking & friends)

| Symptom you might see later in training | Likely cause | What to adjust |
|-----------------------------------------|--------------|----------------|
| Spins in place forever | Progress too weak / time penalty too weak | Increase progress or success weight |
| Camps near goal without finishing | Progress dense reward > success incentive | Stronger `reached_goal`, terminate on success |
| Twitchy steering | Oscillation weight too small | Increase oscillation penalty |
| Never reverses but gets stuck | Reverse penalty too strong | Reduce reverse weight |
| Ignores obstacles | Collision weight too small or sensor not filtering obstacles | Fix contact sensor + weight |
| Explodes / NaNs | Unbounded rewards or bad resets | Normalize terms, check episode length scaling |

This debugging loop is a core **industry skill** in applied RL.

---

## 5.9 What you should do in order (checklist)

1. Read §5.0–5.2 until you can explain reward vs return vs termination out loud.  
2. Create `mdp/rewards.py` and paste the code; skim each docstring.  
3. Create `mdp/terminations.py` and paste.  
4. Create `rewards_cfg.py` and `terminations_cfg.py` under `envs/navigation/config/`.  
5. Write 5 bullets in your notes: “Signals I am rewarding / punishing and why.”  
6. **Do not expect** a training curve yet — Ch.6 (world) and Ch.7 (PPO) still missing.

Optional (Mac): you may add pure-torch unit tests later for shaping formulas with fake distances; not required to proceed.

---

## 5.10 Looking ahead

- **Chapter 6** builds the Mars terrain + goal sampler that make `command_name="target_pose"` real.  
- **Chapter 7** hooks these configs into an env and RSL-RL PPO so the grades start changing the policy.

---

## 5.11 North star

**Observations are senses, actions are intents, rewards are the gradebook, terminations are the bell that ends the exam period — PPO is the student that studies thousands of exams in parallel.**
