# Chapter 1 — Software Architecture & Project Blueprint

> **Status:** Design-only (no training code yet)  
> **Stack target:** Isaac Sim 6.0 · Isaac Lab 3.0 · RSL-RL (PPO) · Perseverance only  
> **Dev workflow:** Author on MacBook M4 → push GitHub → train on NVIDIA machine  

This chapter freezes the *shape* of the project before we write MDP functions, USD assets, or training scripts. Later chapters implement each box in the diagram below.

---

## 1.1 What we are building

**Task (one sentence):** Train a policy that drives Perseverance across procedurally generated Mars-like terrain to a randomly sampled goal while avoiding collisions and staying upright-ish — using only onboard-style sensing.

**Not the goal (yet):** Full Autonav stack, Earth-in-the-loop planning, or multi-rover fleets. One robot, one navigation MDP, one PPO trainer.

### Design principles

| Principle | Why it matters for this rover |
|-----------|-------------------------------|
| **Manager-based MDP** | Isaac Lab splits observations, rewards, actions, terminations into composable terms — same pattern as [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab), but scoped to Perseverance only. |
| **Low-dimensional actions first** | `[v, ω]` (linear + angular velocity) keeps PPO sample-efficient; rocker-bogie + 6 wheels are handled by a *low-level* wheel mapper, not by the policy. |
| **Privileged → onboard curriculum** | Phase A uses height scans / sim state. Phase B adds HazCam/NavCam-derived features. Phase C (optional) adds reach-avoid safety filters. |
| **Docs ↔ code lockstep** | Each chapter owns a folder; we do not implement a module until its chapter exists. |
| **Mac is for structure; GPU is for physics** | Configs, math, docs, Blender→USD prep on M4. Isaac Sim training only on the NVIDIA box. |

---

## 1.2 Advice on your proposed framework (adjustments)

Your breakdown is already close to a production Isaac Lab env. Below are the adjustments I recommend given Perseverance’s rocker-bogie + camera suite, and what RLRoverLab learned the hard way.

### Keep (strong ideas)

1. **Action = `[v, ω]`** — Elegant and correct for navigation. Isaac (or our action term) converts commands into coordinated wheel speeds. Do **not** start with 6 independent wheel torques; that explodes the action space and fights the suspension kinematics.
2. **Weighted multi-term reward** — Progress, goal bonus, collision, smoothness, time, reverse — same family RLRoverLab uses successfully.
3. **Episode ends** — Collision, goal, timeout, out-of-bounds; add “stuck” later; defer “flipped” until IMU pitch/roll thresholds are trusted.
4. **Random reachable goals** — Essential for generalization. Prefer sampling on the *terrain mesh* (not uniform XY in empty space).

### Change or sequence carefully

| Your idea | Recommendation |
|-----------|----------------|
| Feed **camera embeddings** into PPO from day one | **Defer.** Start with a **height scan** (ray caster / elevation grid under and ahead of the rover) + proprioception. Cameras are high-dimensional and slow to learn; treat vision as Phase B. |
| Separate “Observation Builder” vs “State” | In RL jargon for this project: the policy sees an **observation** `o_t`. The simulator has a fuller **state** `s_t`. We may use privileged state for rewards/terminations, but the network only gets `o_t`. |
| Absolute robot `(x,y)` in the observation | Prefer **goal-relative** features in the body frame: distance-to-goal, heading-to-goal, height map in robot frame. Absolute pose hurts generalization across map origins. |
| Terrain slope + roughness as hand features | Derive them from the height scan (local plane fit / variance) instead of inventing parallel sensors early. |
| Energy / torque reward early | Nice physically, but noisy. Start with **action-rate (jerk) penalty**; add torque/energy once wheel drives are stable. |
| “PPO won’t process images; we preprocess distance then feed PPO” | Correct instinct for Phase A. That preprocessing *is* part of the observation pipeline (stereo → depth/occupancy → features). Later you can train an encoder end-to-end if needed. |

### Rocker-bogie specific notes

- **Independent wheels ≠ independent RL actions.** Mechanically each wheel can be driven; for learning we still command chassis twist `[v, ω]` (and optionally a simple steer mode later). A **differential / skid-style mapper** (or Ackermann-like mapper if you add steering articulation) turns twist into six wheel velocity targets.
- **Suspension is passive.** Do not put rocker/bogie joint torques in the action space for v1. Let PhysX resolve the passive joints; observe chassis pitch/roll if useful.
- **HazCams (4 front + 2 rear) and NavCams (mast stereo)** — Model camera prims in USD for realism and Phase B. For Phase A training throughput, a **forward height scanner** (and optional rear) approximates “hazard awareness” without rendering.

### Reach-avoid: how pathing fits (important mental model)

**PPO does not compute a geometric path** the way A\* or RRT\* does. The neural policy is a **reactive controller**:

\[
a_t \sim \pi_\theta(a \mid o_t)
\]

“Best path” emerges *implicitly* if rewards punish collisions and reward progress. That is enough for v1.

**Reach-avoid** (reach goal set \(\mathcal{G}\), avoid unsafe set \(\mathcal{U}\)) is a *formal* framing you can layer on later:

| Layer | Role | When |
|-------|------|------|
| **Shaped rewards + terminations** | Soft reach-avoid via learning | Phase A (now) |
| **Waypoint / subgoal curriculum** | Break long drives into shorter reach problems | Phase A–B |
| **Classical planner + RL tracker** | RRT\*/A\* proposes path; policy tracks | Optional |
| **Safety filter (CBF / HJ)** | Policy proposes \(a\); filter projects to safe set | Advanced |

We will develop the math chapter-by-chapter. For Chapter 1, remember: **policy = stochastic mapping from observation to action; path = what you see when you roll that policy out.**

---

## 1.3 System architecture (logical)

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Training Host (NVIDIA)                    │
│  Isaac Sim 6.0  +  Isaac Lab 3.0  +  RSL-RL PPO                  │
└─────────────────────────────────────────────────────────────────┘
                ▲ configs / checkpoints via Git + artifacts
┌───────────────┴─────────────────────────────────────────────────┐
│                     Repo: Mars_rover_rl (this)                   │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ Assets       │   │ Scene        │   │ MDP Managers         │ │
│  │ Perseverance │──▶│ Terrain      │──▶│ Observations         │ │
│  │ USD / URDF   │   │ Obstacles    │   │ Actions [v, ω]       │ │
│  │ Cameras      │   │ Goals        │   │ Rewards              │ │
│  └──────────────┘   └──────────────┘   │ Terminations         │ │
│                                         │ Events / resets      │ │
│                                         └──────────┬───────────┘ │
│                                                    │             │
│                                         ┌──────────▼───────────┐ │
│                                         │ ManagerBasedRLEnv    │ │
│                                         │ (vectorized envs)    │ │
│                                         └──────────┬───────────┘ │
│                                                    │             │
│                                         ┌──────────▼───────────┐ │
│                                         │ RSL-RL PPO           │ │
│                                         │ Actor-Critic nets    │ │
│                                         └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Your 11 blocks → repo modules

| # | Your block | Lives in |
|---|------------|----------|
| 1 | Rover asset (URDF/USD) | `source/mars_rover_rl/assets/robots/perseverance/` |
| 2 | Mars terrain generator | `source/mars_rover_rl/assets/terrains/mars/` |
| 3 | Obstacle generator | terrain / scene cfg + event terms |
| 4 | Goal generator | `commands` / goal sampler in env cfg |
| 5 | Sensor system | scene sensors + `sensors/` helpers |
| 6 | Observation builder | `envs/navigation/mdp/observations.py` |
| 7 | Action interface | `mdp/actions/` (twist → wheels) |
| 8 | Reward system | `envs/navigation/mdp/rewards.py` |
| 9 | Episode manager | terminations + events in env cfg |
| 10 | PPO trainer | `scripts/train.py` + `configs/agents/` (RSL-RL) |
| 11 | Eval & visualization | `scripts/play.py`, notebooks, Isaac viewport |

---

## 1.4 Repository layout (what each path is for)

```text
Mars_rover_rl/
├── README.md                          # Quick start + links to chapters
├── pyproject.toml                     # Package metadata (added in Ch.2)
├── .gitignore
│
├── docs/
│   ├── README.md                      # Documentation map
│   ├── glossary/                      # RL + robotics jargon cheat sheet
│   ├── assets/                        # Diagrams for docs only
│   └── chapters/
│       ├── 01_software_architecture.md  ← you are here
│       ├── 02_install_and_tooling.md    (planned)
│       ├── 03_rover_asset_pipeline.md   (planned)
│       ├── 04_observation_action_spaces.md
│       ├── 05_rewards_and_terminations.md
│       ├── 06_terrain_goals_obstacles.md
│       ├── 07_ppo_training_loop.md
│       ├── 08_vision_and_reach_avoid.md
│       └── 09_evaluation.md
│
├── source/mars_rover_rl/              # Installable Python package
│   ├── assets/
│   │   ├── robots/perseverance/       # USD, articulation cfg, wheel map
│   │   ├── terrains/mars/             # Procedural / imported Mars meshes
│   │   └── textures/
│   ├── envs/
│   │   └── navigation/
│   │       ├── config/                # Env cfg dataclasses (Isaac Lab style)
│   │       └── mdp/                   # obs, rewards, terminations, events
│   ├── mdp/
│   │   └── actions/                   # Shared action terms: [v,ω] → wheels
│   ├── sensors/                       # Camera / scan helpers (Phase B+)
│   └── utils/
│
├── configs/
│   ├── agents/                        # RSL-RL PPO hyperparams (yaml/py)
│   └── env/                           # Named env presets (easy/hard terrain)
│
├── scripts/
│   ├── train.py                       # Entry: create env + RSL-RL runner
│   ├── play.py                        # Load checkpoint, visualize
│   └── zero_agent.py                  # Sanity: env steps with zero actions
│
├── tests/                             # Unit tests for reward/obs math (CPU)
├── notebooks/                         # Analysis only (optional)
└── logs/                              # Local runs (gitignored)
```

### File responsibilities (when we implement them)

| File (planned) | Responsibility |
|----------------|----------------|
| `assets/robots/perseverance/perseverance.py` | `ArticulationCfg`: links, drives, initial pose |
| `mdp/actions/twist_to_wheels.py` | Map `[v, ω]` → 6 wheel velocity targets |
| `envs/navigation/mdp/observations.py` | Each observation *term* (distance, heading, scan, …) |
| `envs/navigation/mdp/rewards.py` | Each reward *term* (progress, collision, …) |
| `envs/navigation/mdp/terminations.py` | Done flags |
| `envs/navigation/mdp/events.py` | Resets, domain randomization |
| `envs/navigation/config/navigation_env_cfg.py` | Wires scene + MDP managers together |
| `configs/agents/rsl_rl_ppo_cfg.py` | Network size, γ, λ, clip ε, learning rate |
| `scripts/train.py` | CLI: `--num_envs`, `--max_iterations`, seed |

Empty package folders already exist so Chapter 2+ can drop files into a known tree.

---

## 1.5 MDP sketch (preview — detailed in Ch.4–5)

### Observation (policy input) — Phase A proposal

Concatenate into one vector (Isaac Lab `ObservationGroupCfg`):

| Term | Meaning | Notes |
|------|---------|-------|
| `distance_to_goal` | \(\|p_g - p_r\|_{xy}\) | Scale/normalize |
| `heading_to_goal` | Angle in body frame | \(\operatorname{atan2}\) |
| `base_lin_vel` / `base_ang_vel` | Chassis twist | From articulation data |
| `last_action` | Previous `[v, ω]` | Helps smoothness |
| `height_scan` | Local elevation grid | Stand-in for HazCam awareness |

**Later:** wheel speeds, IMU roll/pitch, camera encoder features.

### Action

\[
a_t = [v_t,\ \omega_t] \in [v_{\min}, v_{\max}] \times [\omega_{\min}, \omega_{\max}]
\]

Clipped and possibly rate-limited; low-level controller tracks wheel speeds.

### Reward (conceptual)

\[
r_t = w_g r^{\text{goal}} + w_p r^{\text{progress}} - w_c r^{\text{collision}}
      - w_s r^{\text{smooth}} - w_u r^{\text{unsafe}} - w_t r^{\text{time}} - w_r r^{\text{reverse}}
\]

Exact weights and functional forms are Chapter 5. Inspiration (not copy) from RLRoverLab: inverse-distance shaping, oscillation penalty, reverse soft constraint, contact-force collision.

### Terminations

`success` · `collision` · `timeout` · `out_of_bounds` · (later) `stuck` · (later) `flipped`

---

## 1.6 Training vs authoring machines

| Activity | MacBook M4 | NVIDIA laptop |
|----------|------------|---------------|
| Edit Python / docs / configs | ✅ | ✅ |
| Blender geometry polish → export | ✅ | optional |
| Unit-test pure torch reward math | ✅ | ✅ |
| Isaac Sim / Lab import USD | ❌ (no full GPU Sim path we rely on) | ✅ |
| Multi-env PPO training | ❌ | ✅ |
| Play / record videos | ❌ | ✅ |

**Install reality check:** Isaac Lab 3.0 + Sim 6.0 are installed on the NVIDIA machine. This repo stays *thin*: our package depends on Isaac Lab APIs but does not vendor Omniverse.

---

## 1.7 Planned documentation chapters

| Ch | Title | Delivers |
|----|-------|----------|
| **1** | Software architecture | This document + folder skeleton |
| **2** | Install & tooling | `pyproject.toml`, Isaac Lab editable install notes, RSL-RL |
| **3** | Rover asset pipeline | Blend → USD, joints, cameras, wheel frames |
| **4** | Observation & action spaces | Formal dims, normalization, action term |
| **5** | Rewards & terminations | Equations + implementation |
| **6** | Terrain, goals, obstacles | Procedural Mars + sampling |
| **7** | PPO training loop | RSL-RL runner, logging, checkpoints |
| **8** | Vision & reach-avoid | HazCam/NavCam features + safety math |
| **9** | Evaluation | Metrics, ablations, failure modes |

---

## 1.8 Subchapter — PPO for Perseverance (gentle but precise)

You already know **TD** and **Q-learning**. PPO is the workhorse we will use with continuous actions `[v, ω]`. Here is the bridge.

### 1.8.1 From Q-learning to policy gradients (why PPO exists)

- **Q-learning** learns \(Q(s,a)\): “how good is this action?” For discrete actions (left/right), you pick \(\arg\max_a Q\).
- Our actions are **continuous** (any speed/turn in a range). Maximizing \(Q\) over a continuum is awkward.
- **Policy gradient** methods instead learn a parameterized policy \(\pi_\theta(a\mid o)\) directly — usually a Gaussian whose mean (and sometimes std) is output by a neural net.

**Jargon:**  
- **Actor** = the policy network \(\pi_\theta\).  
- **Critic** = a value network \(V_\phi(o)\) that estimates “how good is this observation?” (like a state-value baseline).  
- **Advantage** \(A_t\) ≈ “was this action better or worse than expected?”

### 1.8.2 The RL loop in our project

At each step \(t\):

1. Env builds observation \(o_t\) (distance, heading, height scan, …).
2. Actor samples \(a_t = [v,\omega] \sim \pi_\theta(\cdot\mid o_t)\).
3. Action term converts \(a_t\) → wheel commands; PhysX steps.
4. Reward \(r_t\) and next obs \(o_{t+1}\) (or terminal).
5. Store transition in a **rollout buffer**.
6. After enough steps across many parallel envs, **update** \(\theta\) and \(\phi\).

Isaac Lab runs **thousands of envs in parallel** on GPU — that is why RSL-RL + Isaac is so effective for locomotion/navigation.

### 1.8.3 Return, value, advantage (TD family you know)

**Discounted return** from time \(t\):

\[
R_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}
\]

\(\gamma \in (0,1)\) (e.g. 0.99) says “prefer sooner reward.”

**Critic** learns \(V_\phi(o_t) \approx \mathbb{E}[R_t \mid o_t]\).

**TD error** (one-step), familiar from TD learning:

\[
\delta_t = r_t + \gamma V_\phi(o_{t+1}) - V_\phi(o_t)
\]

**GAE** (Generalized Advantage Estimation) blends multi-step TD errors into a stabler advantage \(\hat{A}_t\). You do not need to derive GAE to use it — RSL-RL computes it — but the intuition is: *advantage tells the actor which actions to reinforce.*

### 1.8.4 The PPO objective (the “clip” trick)

Vanilla policy gradient pushes \(\theta\) to increase probability of actions that had positive advantage. If one update is too large, the policy can collapse.

**PPO** constrains the update using a probability ratio:

\[
\rho_t(\theta) = \frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta_{\text{old}}}(a_t\mid o_t)}
\]

**Clipped surrogate** (core PPO idea):

\[
L^{\text{CLIP}}(\theta) =
\mathbb{E}_t\Big[
\min\big(
\rho_t\hat{A}_t,\
\operatorname{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t
\big)
\Big]
\]

- If \(\hat{A}_t > 0\) (good action): we may increase its probability, but not by more than factor \(1+\varepsilon\).
- If \(\hat{A}_t < 0\) (bad action): we may decrease it, but not too aggressively.

Typical \(\varepsilon \approx 0.2\).

**Total loss** (schematic):

\[
L = - L^{\text{CLIP}} + c_v \|V_\phi - R\|^2 - c_e\, \mathcal{H}[\pi_\theta]
\]

- Value loss: train the critic.  
- Entropy bonus \(\mathcal{H}\): keep exploring (important early so the rover does not freeze in one steering habit).

### 1.8.5 How “best action” is chosen at run time

During **training**, actions are **sampled** (exploration).  
During **evaluation / deployment**, we usually take the **mean** of the Gaussian (deterministic drive).

There is no separate “path optimizer” inside PPO. If you want an explicit path:

- either the **observation** already encodes a local cost map and the policy learns to follow low-cost corridors, or  
- you add a **planner** outside the policy (Chapter 8).

### 1.8.6 Mapping PPO knobs to this rover

| Knob | Role for Perseverance nav |
|------|---------------------------|
| `num_envs` | Parallel Mars instances — more = faster data |
| `gamma` | Long horizons to reach distant goals |
| `gae_lambda` | Bias-variance of advantage |
| `clip_param` ε | Stability of policy updates |
| `entropy_coef` | Exploration of turns vs straight drives |
| `learning_rate` | Often scheduled / adaptive in RSL-RL |
| Network MLP size | Must fit observation dim (scan can be large) |

### 1.8.7 Mini glossary for this subchapter

| Term | Plain meaning |
|------|----------------|
| **MDP** | Rules of the game: states/obs, actions, rewards, transitions |
| **Policy π** | The driver’s brain (neural net) |
| **Rollout** | A stretch of experience collected before an update |
| **On-policy** | PPO learns from data collected by the *current* policy (then discards) |
| **Sample efficiency** | How much sim time you need before the rover stops being terrible |
| **Domain randomization** | Randomize friction, terrain, noise so the policy generalizes |
| **Privileged information** | Sim-only signals used in reward/training but not on the real rover |

---

## 1.9 What we take from RLRoverLab (inspiration only)

Useful patterns to mirror:

- Manager-based `RoverEnvCfg` with separate Obs / Rewards / Terminations / Commands.
- Height scanner as primary local terrain observation.
- Goal as a **command** resampled on terrain.
- Oscillation + reverse penalties for smooth driving.
- Contact sensor filtered to obstacle prims for collision.

What we intentionally do differently:

- **Single robot:** Perseverance only (no multi-rover registry).
- **Isaac Sim 6 / Lab 3** (their badges show older versions — APIs may differ; we follow current Lab 3 docs).
- **RSL-RL first** (they emphasize skrl in places; Isaac Lab’s native RSL-RL path is our default).
- **Documented phased vision / reach-avoid** instead of a generic multi-task suite.

---

## 1.10 Chapter 1 checklist

- [x] Problem statement and design principles  
- [x] Framework advice (what to keep / defer)  
- [x] Logical architecture diagram  
- [x] Concrete repo tree and file roles  
- [x] MDP preview  
- [x] Mac vs NVIDIA workflow  
- [x] PPO primer tied to this project  
- [ ] Chapter 2: install notes + `pyproject.toml` (next)

---

## 1.11 Next step

**Chapter 2 — Install & tooling:** pin how this package sits beside Isaac Lab 3.0, RSL-RL config layout, and what you can verify on the Mac before cloning to the NVIDIA machine.

When you are ready, say the word and we write Chapter 2 (still light on physics code) and add `pyproject.toml` + README quickstart stubs.
