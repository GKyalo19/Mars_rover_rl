# Chapter 8 — Vision & Reach-Avoid

> **How we work**
> - Explanations + full code live **in this chapter**.
> - **You** create files and paste while reading.
> - I will **not** write into your `source/` package unless you ask.
>
> **Goal:** Finish the `height_scan` observation Ch.4 reserved, then (optionally) add HazCam/NavCam-style camera vision and a lightweight **reach-avoid** safety layer on top of your trained policy.
> **Depends on:** Ch.4 (obs plumbing), Ch.6 (scene already has a `height_scanner` ray caster + `robot` slot), Ch.7 (a PPO loop that actually runs).
> **Hard gate (same as Ch.7, plus one more):** Training/vision only run for real once `MarsNavSceneCfg.robot` is a real `ArticulationCfg`. If you want the camera track, your USD also needs HazCam/NavCam camera prims (Ch.3/3b) — those are currently placeholders.
> **Optional chapter:** Unlike Ch.4–7, Chapter 8 is **not required** to have a working nav policy. Track A (height scan) is cheap and worth doing; Tracks B (vision) and C (reach-avoid) are Phase B/C polish — read them now, implement whenever you're ready.
> **Next:** Ch.9 evaluation / play.

---

## 8.0 Bigger picture — three optional upgrades, not one big feature

Chapter 1 already sketched this as a **curriculum**, not a single leap:

| Layer | What it adds | Phase |
|-------|-------------|-------|
| **Privileged → onboard curriculum** | Phase A uses height scans / sim state → Phase B adds HazCam/NavCam-derived features → Phase C (optional) adds reach-avoid safety filters | Ch.1 §1.1 |
| **Reach-avoid framing** | Shaped rewards (Phase A, done in Ch.5) → waypoint/subgoal curriculum (Phase A–B) → classical planner + RL tracker (optional) | Ch.1 §1.2 |

Chapter 8 turns those two roadmap rows into three concrete, independent tracks:

```text
Track A — Height scan            (finishes a Ch.4 promise, cheap, no new assets)
Track B — Camera vision          (HazCam/NavCam images → CNN-fused policy, heavier)
Track C — Reach-avoid shield     (safety layer wrapped around a trained policy)
```

You can do **A only**, **A + B**, or **A + C** — they don't require each other except that B and C both assume A's height scan already works, and C assumes you have *something* trained from Ch.7.

```text
Ch.4 reserved height_scan
        │
        ▼
Track A: wire the real ray-caster term   ◀── do this first, always
        │
        ├──▶ Track B: add HazCam/NavCam TiledCamera + CNN-fused actor-critic
        │
        └──▶ Track C: reach-avoid shield wrapped around whichever policy you trained
```

### Why this is its own chapter and not folded into Ch.4/Ch.7

Height scan needed the scene (Ch.6) to exist first — that's why it was a stub in Ch.4. Vision and reach-avoid are **substantially more expensive** (rendering cost, new network architecture, new safety math) than anything before, so they get isolated, honest treatment rather than sneaking complexity into "just one more observation term."

---

## 8.1 Jargon, explained properly

### Height scan (finishing the Ch.4 stub)

A **height scan** is a small grid of ray-cast distances/heights sampled under and ahead of the robot — Ch.6's `height_scanner` `RayCasterCfg` already casts a `3.0 × 3.0` m grid at `0.2` m resolution beneath `base_link`. Track A just reads that sensor's buffer into an observation term. It is **not** an image; it's a flat vector of floats, cheap for an MLP, and does not need rendering (`RayCaster` uses PhysX raycasts, not the render pipeline).

### RGB vs depth vs semantic camera

| Kind | Returns | Use here |
|------|---------|----------|
| **RGB** | 3-channel color image | Closest to a real HazCam feed; hardest to learn from (lighting, texture noise) |
| **Depth** | 1-channel distance-per-pixel | Often easier for navigation — directly geometric, less nuisance variation |
| **Semantic / segmentation** | Per-pixel class id | Overkill for Phase B; mentioned for completeness |

We start with **depth** for Track B: it is the most learnable signal for "don't hit that" and is closer in spirit to the height scan you already have.

### TiledCamera

Isaac Lab's `TiledCamera` is the render-based camera sensor built **for RL** — it batches every parallel env's camera into one big tiled render pass instead of one draw call per env, which is what makes camera observations tractable at `num_envs=64+`. Plain `Camera` sensors exist too but don't scale the same way; prefer `TiledCameraCfg` for training.

### Observation groups and `concatenate_terms`

Ch.4's `ObservationsCfg.PolicyCfg` set `concatenate_terms = True` — every term gets stacked into one flat vector, which only works if every term is 1D-per-env. An image term is **not** 1D (it's `(N, C, H, W)`), so a vision-aware group must set `concatenate_terms = False` and hand the policy a **dict** of tensors instead: `{"vector": Tensor[N, n], "depth_cam": Tensor[N, C, H, W]}`.

### CNN encoder / feature fusion

A small **CNN encoder** compresses the depth image into a fixed-size feature vector, which is then **fused** (concatenated) with the vector observation (distance/heading/height-scan/etc.) before the actor/critic MLP heads. This is the standard "multi-modal obs" pattern — vision networks almost never receive raw pixels straight into an MLP.

### Asymmetric actor-critic

Nothing stops the **critic** from seeing more than the **actor**. A common trick: give the critic privileged/extra info (e.g. true distance-to-nearest-obstacle, exact terrain height) that a real rover couldn't sense, while the actor only gets what would be available onboard. This tightens the value estimate without breaking sim-to-real honesty of the deployed policy. We mention it here because it's the natural next step once you have separate obs groups — Phase A used one shared group; Track B is where the split starts paying off.

### Reach-avoid (formal)

**Reach-avoid** is a control-theory framing: given a **reach set** \(\mathcal{G}\) (the goal region) and an **avoid set** \(\mathcal{U}\) (unsafe region — rocks, cliffs, out-of-map), find a controller that drives the system into \(\mathcal{G}\) while never entering \(\mathcal{U}\), for all time. Formally computing the exact set of states from which this is guaranteed possible is **Hamilton-Jacobi (HJ) reachability** — a PDE solved over the state space. That's a research-grade tool (see `hj_reachability`, `optimized_dp`) and **out of scope** for this project; we mention it so the vocabulary is correct.

### Safety filter / shield / control barrier function (CBF)

A **shield** (a.k.a. **safety filter**) is a lightweight practical stand-in for full HJ reachability: at each step, check whether the policy's proposed action is "obviously" heading into the avoid set (e.g. nearest obstacle distance from the height scan is below a threshold **and** the commanded velocity points toward it); if so, override or clip the action. A **control barrier function (CBF)** formalizes "how close to the danger boundary am I" as a scalar function whose sign tells you safe vs unsafe — a shield is essentially a CBF check plus an override rule. Track C implements the simple version of this, not the full CBF optimization machinery.

### Backup policy

The action a shield substitutes **instead of** the policy's unsafe suggestion — often just "stop" (`v=0, ω=0`) or "turn away from nearest obstacle." Simple, unglamorous, effective.

---

## 8.2 Track A — finish the height scan (do this one first)

### Why now, not Ch.4

Ch.4 said: *"height_scan / base velocities land in a later pass once the scene sensors and articulation cfg exist."* Ch.6 built `height_scanner` (a `RayCasterCfg` on `base_link`, `3×3` m grid, `0.2` m resolution → **15×15 = 225** ray hits). Now the sensor exists, so the stub can become real.

### File: update `mars_rover/mdp/observations.py`

Add this function alongside `distance_to_goal` / `heading_to_goal` / `last_action`:

```python
def height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg_name: str = "height_scanner",
    offset: float = 0.5,
) -> torch.Tensor:
    """Flattened local elevation grid under/ahead of the rover, shape (N, 225).

    Puzzle piece: a cheap, always-available stand-in for HazCam awareness.
    Values are hit-distance-below-sensor; ``offset`` re-centers them so "flat
    ground directly under the sensor" reads close to zero instead of a large
    constant (the sensor sits ``offset`` m above the chassis per Ch.6's
    ``RayCasterCfg.OffsetCfg``).

    Requires: MarsNavSceneCfg.height_scanner (Ch.6) attached to the scene.
    """
    sensor = env.scene.sensors[sensor_cfg_name]
    # ray_hits_w: world-frame hit points, shape (N, num_rays, 3).
    hit_heights = sensor.data.ray_hits_w[..., 2]
    sensor_height = sensor.data.pos_w[:, 2].unsqueeze(-1)
    relative_height = sensor_height - hit_heights - offset
    return relative_height
```

| Piece | Plain English |
|-------|----------------|
| `env.scene.sensors[...]` | How Lab exposes any scene sensor by its cfg attribute name |
| `ray_hits_w[..., 2]` | Just the Z (height) component of each of the 225 hit points |
| `sensor_height - hit_heights` | "How far below the sensor is the ground at this ray" — bigger means a dip, smaller/negative means a bump |
| `offset` | Cancels out the sensor's own mount height so flat ground reads ≈ 0 |

### File: update `envs/navigation/config/observations_cfg.py`

```python
height_scan = ObsTerm(
    func=mdp.observations.height_scan,
    params={"sensor_cfg_name": "height_scanner", "offset": 0.5},
    scale=1.0,
    clip=(-2.0, 2.0),  # guard against a stray ray hitting something far away
)
```

Add this line inside `PolicyCfg`, alongside the existing three terms. Observation dimension jumps from 4 to **229** (`distance` 1 + `heading` 1 + `last_action` 2 + `height_scan` 225) — bump `actor_hidden_dims` / `critic_hidden_dims` in `PerseverancePPORunnerCfg` (Ch.7) if training looks underpowered once this lands; `[256, 128, 64]` should still be fine as a first try.

**Pass criteria (once robot is real):** `zero_agent.py` (Ch.7 §7.7) still runs; printed reward stays finite; no shape errors from the observation manager.

---

## 8.3 Track B — HazCam/NavCam vision (optional, heavier)

### The honest cost callback

This directly changes the headless-vs-rendering conversation from earlier: **once a camera is in the observation space, you can no longer dodge rendering with `--headless`.** `--headless` only skips the *viewport window*; it does not skip the *camera sensor's render pass* that `TiledCamera` needs to produce pixels every step your policy asks for one. Camera-based RL pays a rendering cost on every single training step, not just during `--video` windows. This is the real reason Ch.1 deferred vision to Phase B: it's a genuine, permanent throughput cost, not a viewport-vanity cost. Expect meaningfully lower steps/second than Track A once this is wired in — budget for it (smaller `num_envs`, smaller image resolution, or more wall-clock patience).

### File: update `mars_rover/assets/terrains/mars/mars_scene_cfg.py`

Add one camera to `MarsNavSceneCfg` (start with a single forward HazCam-style depth camera; add a rear/NavCam later the same way):

```python
from isaaclab.sensors import TiledCameraCfg

# ---- forward hazard camera (depth) ----
hazcam_front = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link/hazcam_front",  # match your USD camera prim
    offset=TiledCameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.3), rot=(0.0, 0.0, 0.0, 1.0)),  # Lab 3.0 (x, y, z, w) identity
    data_types=["distance_to_camera"],  # depth; add "rgb" later if you want color too
    spawn=None,  # camera prim already exists in the USD (Ch.3/3b); don't re-spawn it
    width=64,
    height=64,
    update_period=0.0,  # match sim rate; tune if render becomes the bottleneck
)
```

| Field | Meaning | Why small numbers |
|-------|---------|--------------------|
| `width=64, height=64` | Low-res on purpose | CNNs learn fine at low res for "is something close" tasks; every doubled dimension roughly quadruples render + conv cost |
| `data_types=["distance_to_camera"]` | Depth channel | Cheapest to learn from, closest to height-scan intuition |
| `spawn=None` | Reuse the USD's camera prim | If your Perseverance USD (Ch.3) doesn't have a camera prim yet, this is part of the vision hard gate — add one in Blender/USD first |

### File: update `envs/navigation/config/observations_cfg.py` (new group)

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        distance_to_goal = ObsTerm(func=mdp.observations.distance_to_goal, params={"command_name": "target_pose"}, scale=0.1)
        heading_to_goal = ObsTerm(func=mdp.observations.heading_to_goal, params={"command_name": "target_pose"}, scale=1.0 / math.pi)
        last_action = ObsTerm(func=mdp.observations.last_action)
        height_scan = ObsTerm(func=mdp.observations.height_scan, params={"sensor_cfg_name": "height_scanner", "offset": 0.5}, clip=(-2.0, 2.0))

        def __post_init__(self) -> None:
            self.concatenate_terms = True  # this group is still one flat vector
            self.enable_corruption = False

    @configclass
    class VisionCfg(ObsGroup):
        """Kept separate from ``policy`` because an image can't be flattened
        into the same vector without a CNN in between."""

        depth_cam = ObsTerm(func=mdp.observations.hazcam_depth, params={"sensor_cfg_name": "hazcam_front"})

        def __post_init__(self) -> None:
            self.concatenate_terms = False  # keep image shape intact for the CNN

    policy: PolicyCfg = PolicyCfg()
    vision: VisionCfg = VisionCfg()
```

### File: add to `mars_rover/mdp/observations.py`

```python
def hazcam_depth(env: ManagerBasedRLEnv, sensor_cfg_name: str = "hazcam_front") -> torch.Tensor:
    """Forward depth image, shape (N, H, W, 1) → normalized to roughly [0, 1].

    Puzzle piece: raw depth is in meters and can be large/unbounded (sky,
    open ground). Clamp + normalize so the CNN sees a friendly numeric range.
    """
    sensor = env.scene.sensors[sensor_cfg_name]
    depth = sensor.data.output["distance_to_camera"]
    depth = torch.clamp(depth, min=0.0, max=10.0) / 10.0
    return depth
```

### CNN-fused actor-critic — full code (advanced, version-sensitive)

RSL-RL's default `ActorCritic` expects one flat observation vector, not a dict with an image inside. To use both groups, you need a **custom actor-critic class** with a CNN branch. This is the most Lab/RSL-RL-version-sensitive code in this project — check your installed `rsl_rl` version's `ActorCritic` base class signature before pasting blindly.

**File:** `mars_rover/envs/navigation/agents/vision_actor_critic.py`

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom RSL-RL actor-critic that fuses vector obs + a depth image.

Educational reference implementation — verify against your installed
rsl_rl.modules.ActorCritic before relying on it for a real run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_nn_activation


class DepthEncoder(nn.Module):
    """Small CNN: (N, H, W, 1) depth image -> (N, feature_dim)."""

    def __init__(self, height: int = 64, width: int = 64, feature_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 4 * 4, feature_dim)

    def forward(self, depth_img: torch.Tensor) -> torch.Tensor:
        # (N, H, W, 1) -> (N, 1, H, W) for Conv2d.
        x = depth_img.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class VisionActorCritic(ActorCritic):
    """Fuses ``vision.depth_cam`` with ``policy`` vector obs before the
    standard actor/critic MLP heads.

    Kept intentionally close to RSL-RL's own ``ActorCritic`` so the runner's
    checkpointing / logging code paths keep working unmodified.
    """

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, image_hw=(64, 64), **kwargs):
        # Reserve room in the MLP input size for the CNN's fused features.
        feature_dim = 64
        super().__init__(
            num_actor_obs=num_actor_obs + feature_dim,
            num_critic_obs=num_critic_obs + feature_dim,
            num_actions=num_actions,
            activation=kwargs.pop("activation", "elu"),
            **kwargs,
        )
        self.depth_encoder = DepthEncoder(*image_hw, feature_dim=feature_dim)

    def _fuse(self, vector_obs: torch.Tensor, depth_img: torch.Tensor) -> torch.Tensor:
        img_features = self.depth_encoder(depth_img)
        return torch.cat([vector_obs, img_features], dim=-1)

    def act(self, vector_obs, depth_img, **kwargs):
        return super().act(self._fuse(vector_obs, depth_img), **kwargs)

    def evaluate(self, vector_obs, depth_img, **kwargs):
        return super().evaluate(self._fuse(vector_obs, depth_img), **kwargs)
```

**File:** `mars_rover/envs/navigation/agents/rsl_rl_ppo_cfg.py` — full file, both runner configs together:

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO settings for Perseverance navigation.

These numbers are *starting points*. Expect to tune after you see TensorBoard curves.

Two runner configs live here:
  - PerseverancePPORunnerCfg        (Ch.7, Phase A  — vector obs only)
  - PerseveranceVisionPPORunnerCfg  (Ch.8 Track B, optional — vector + depth camera)
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PerseverancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase A: vector-observation-only PPO settings (Ch.7)."""

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


@configclass
class PerseveranceVisionPPORunnerCfg(PerseverancePPORunnerCfg):
    """Ch.8 Track B (optional): swaps in the depth-camera-fused actor-critic.

    Inherits every PPO/algorithm knob from PerseverancePPORunnerCfg unchanged —
    only the policy network class and log folder name differ. Requires
    VisionActorCritic (mars_rover/envs/navigation/agents/vision_actor_critic.py)
    to be importable wherever RSL-RL resolves ``class_name``.
    """

    # Separate log folder so vision runs don't mix with Phase A vector runs.
    experiment_name = "perseverance_nav_vision"

    policy = RslRlPpoActorCriticCfg(
        class_name="VisionActorCritic",  # must be importable where RSL-RL looks for it
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    # algorithm block intentionally omitted — inherited unchanged from
    # PerseverancePPORunnerCfg above. Override here later if vision training
    # needs different PPO hyperparameters (e.g. lower learning_rate).
```

**Important honesty note:** whether `class_name` accepts an arbitrary custom class this way (vs. requiring registration elsewhere) depends on your exact `rsl_rl` release. Treat this block as a **starting sketch**, not copy-paste-guaranteed — read `rsl_rl/runners/on_policy_runner.py` in your installed package to confirm how it instantiates the policy class before a real run.

### Wiring caveat: two obs groups into a runner

`RslRlVecEnvWrapper` (Ch.7) by default assumes a single flat `policy` observation. Feeding it two groups (`policy` + `vision`) for the custom class above will likely need either (a) a small wrapper subclass that passes both groups through to `act`/`evaluate`, or (b) checking whether your Lab 3.x `RslRlVecEnvWrapper` already supports dict observations. This is genuinely the frontier of "adapt to your installed version" — flagging it now so you're not surprised later.

---

## 8.4 Track C — reach-avoid shield (optional safety layer)

### Where this sits relative to what you already built

| Layer (from Ch.1 roadmap) | Status |
|---|---|
| Shaped rewards + terminations (soft reach-avoid via learning) | **Done — Ch.5** (`progress_to_goal`, `collision_penalty`, `goal_reached` termination) |
| Waypoint / subgoal curriculum | Not built — future upgrade if long drives are too hard to learn end-to-end |
| Classical planner + RL tracker | Not built — optional, heavier alternative architecture |
| **Explicit safety shield (this track)** | **New in Ch.8** |

A shield does **not** replace the reward-shaped safety Ch.5 already teaches — it's a cheap **runtime backstop** for the cases the policy hasn't fully learned yet (early training, distribution shift, evaluation on a harder map in Ch.9).

### The simple version we're actually building

Not full HJ reachability. A **height-scan-based clip rule**:

1. Look at the height-scan rays roughly in front of the commanded heading.
2. If the minimum clearance there is below a safety threshold **and** the commanded `v > 0` (driving toward it), zero out or shrink `v`.
3. Otherwise, pass the policy's action through unchanged.

### File: `mars_rover/mdp/safety/reach_avoid_shield.py`

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight reach-avoid shield: a runtime backstop, not a formal HJ solver.

Use at play/eval time (Ch.9) or even during training if early rollouts are
too crash-heavy to learn from. Not required for Ch.7's training loop.
"""

from __future__ import annotations

import torch


def shield_action(
    action: torch.Tensor,
    height_scan: torch.Tensor,
    *,
    clearance_threshold: float = 0.15,
    forward_ray_slice: slice = slice(90, 135),  # rays roughly "ahead" in a 15x15 grid
) -> torch.Tensor:
    """Clip forward velocity when the scan says danger is close ahead.

    Args:
        action: Policy output, shape (N, 2) = [v, ω] (already scaled).
        height_scan: Flattened ray-cast grid, shape (N, 225) (Track A term).
        clearance_threshold: Minimum "safe" relative height/clearance (m).
        forward_ray_slice: Which flattened indices count as "ahead" —
            depends on your grid layout/resolution; verify against Ch.8 §8.2.

    Returns:
        Possibly-modified action, same shape as ``action``.
    """
    forward_clearance = height_scan[:, forward_ray_slice].amin(dim=-1)
    unsafe = forward_clearance < clearance_threshold

    safe_action = action.clone()
    driving_forward = action[:, 0] > 0
    override = unsafe & driving_forward
    safe_action[override, 0] = 0.0  # backup policy: stop forward motion
    return safe_action
```

| Piece | Plain English |
|-------|----------------|
| `forward_ray_slice` | Which of the 225 flattened height-scan cells count as "in front" — this is a **grid-layout assumption**; verify indices match your actual `GridPatternCfg` ordering before trusting it |
| `amin(dim=-1)` | Worst-case (closest) clearance among forward rays |
| `override` | Only intervene when both unsafe **and** actually driving into it — don't fight the policy on turns/reverses |
| Backup policy | Here: just stop forward motion (`v=0`), keep `ω` (turning) untouched so the rover can still steer away |

### How you'd actually use it

At play/eval time (Ch.9's `play.py`), after the policy produces an action and before you call `env.step(action)`:

```python
raw_action = policy(obs)
safe_action = shield_action(raw_action, obs["height_scan"])
obs, reward, terminated, truncated, info = env.step(safe_action)
```

**This is optional and does not gate Ch.9.** Evaluate the raw policy first; only add the shield if you observe avoidable collisions you want backstopped.

### Further reading, not required

- **Control barrier functions (CBFs):** formalize the "how close to the boundary" scalar and derive a provably-safe filtering QP — much stronger guarantees than the clip rule above, meaningfully more math/engineering.
- **HJ reachability toolboxes** (`hj_reachability`, `optimized_dp`): compute the actual reach-avoid set over the full state space — research-grade, expensive, usually reserved for lower-dimensional systems than a full navigation MDP.

---

## 8.5 Files map (this chapter)

```text
source/mars_rover_rl/mars_rover/
├── mdp/
│   ├── observations.py             # + height_scan (Track A), + hazcam_depth (Track B)
│   └── safety/
│       └── reach_avoid_shield.py   # Track C
├── assets/terrains/mars/
│   └── mars_scene_cfg.py           # + hazcam_front TiledCameraCfg (Track B)
└── envs/navigation/
    ├── config/
    │   └── observations_cfg.py     # + height_scan term, + VisionCfg group (Track B)
    └── agents/
        ├── vision_actor_critic.py  # Track B custom network
        └── rsl_rl_ppo_cfg.py       # + PerseveranceVisionPPORunnerCfg (Track B)
```

Track A touches 2 files. Track B touches 5 (including the two new ones). Track C adds 1 new file and is otherwise called from wherever you run inference.

---

## 8.6 Realistic expectations (same honesty as Ch.7)

1. **Track A** works the moment the robot articulation is real (Ch.7's hard gate) — no extra assets needed, since `height_scanner` was already fully defined in Ch.6.
2. **Track B** additionally needs actual camera prims in your Perseverance USD (a Ch.3/3b task, not yet done per your repo's asset folders being placeholders), plus verifying your `rsl_rl`/Isaac Lab version's support for custom actor-critic classes and multi-group observations.
3. **Track C** needs *some* trained policy to shield in the first place — it's a Ch.9-adjacent tool, not a training-time requirement.
4. None of this changes Ch.7's fundamentals — PPO still optimizes whatever reward you defined; vision and shields change what the policy can *sense* and what runtime safety net exists, not the learning algorithm itself.

---

## 8.7 Checklist

1. Explain why `height_scan` was a stub in Ch.4 but real in Ch.8.
2. Paste the `height_scan` observation function + wire it into `ObservationsCfg` (Track A).
3. Explain why `--headless` doesn't save you from camera render cost the way it does for the viewport (tie back to Ch.7 §7.9).
4. (Optional) Paste the `hazcam_front` camera cfg, `VisionCfg` group, `hazcam_depth` term, and `VisionActorCritic` (Track B) — only after camera prims exist in your USD.
5. (Optional) Paste `reach_avoid_shield.py` and explain, in your own words, the difference between this clip rule and full HJ reachability (Track C).
6. Write one note: which track(s) you're actually implementing now vs. deferring, and why.

---

## 8.8 Looking ahead to Chapter 9

Chapter 9 will:

- load a checkpoint (Ch.7) and run a dedicated non-headless evaluation rollout
- define success-rate / collision-rate / time-to-goal metrics
- optionally layer in Track C's shield during evaluation to compare shielded vs unshielded failure modes

---

## 8.9 North star

**Vision widens what the rover can sense; a reach-avoid shield narrows what it's allowed to do when sensing says danger — neither replaces the reward-shaped skill PPO already learned in Chapter 7, they extend it.**
