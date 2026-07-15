# Chapter 4 — Observation & Action Spaces

> **Mode:** Full starter code + deep “puzzle piece” explanations.  
> **Goal:** Define what Perseverance **sees** and what it **commands** — the two interfaces between the world and PPO.  
> **Depends on:** Chapter 2 (`import mars_rover`) · Chapters 1 / 3 / 3b concepts (MDP, joints, `[v, ω]`).  
> **Does not finish yet:** Rewards (Ch.5), Mars terrain (Ch.6), `train.py` (Ch.7).

---

## 4.0 Bigger picture — where this chapter sits in the puzzle

You already built the **loading dock** (Ch.2 packaging).  
You are building / built the **physical robot** (Ch.3 / 3b: meshes → USD → joints → drives).  

Now we define the **language** between the simulator and the learning algorithm:

```text
                    ┌──────────────┐
   Mars world ──▶   │ Observation  │ ──▶  vector o_t  ──▶  PPO (actor)
   (sim truth)      │  builder     │                      │
                    └──────────────┘                      ▼
                                                    action a_t = [v, ω]
                                                          │
                    ┌──────────────┐                      ▼
   Wheel drives ◀── │ Action term  │ ◀── twist → 6 wheel speeds
                    └──────────────┘
```


| Puzzle piece             | Chapter | Role                                    |
| ------------------------ | ------- | --------------------------------------- |
| Package `mars_rover`     | 2       | Code can be imported                    |
| Articulated Perseverance | 3b      | Wheels can spin under PhysX             |
| **Observation space**    | **4**   | What the policy is allowed to know      |
| **Action space**         | **4**   | What the policy is allowed to output    |
| Rewards / terminations   | 5       | What “good driving” means numerically   |
| Terrain + goals          | 6       | The exam questions each episode         |
| PPO / RSL-RL             | 7       | How the policy improves from experience |


**Industry parallel:** In robotics you often split **perception → planning/control → actuation**.  
In deep RL for navigation, **perception** is compressed into an observation vector (or images later), **planning/control** is the neural policy, **actuation** is the action term that talks to joint drives. Same sandwich — different filling.

---



## 4.1 Jargon that must click together

Read these as one story, not a dictionary dump.

### MDP (Markov Decision Process)

The formal game:

> At time t, agent sees observation o_t, picks action a_t, environment returns reward r_t and next observation o_{t+1}.

**Markov** (simplified): the observation (or state) should contain enough info that you don’t need the whole past to choose well. That’s why we include things like **last action** and velocities — they carry recent history in a compact way.

### State vs observation (privileged information)


| Term              | Meaning                                                     | Perseverance example                                                           |
| ----------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **State s**       | Full sim truth                                              | Exact rock mesh IDs, true friction                                             |
| **Observation o** | What the **policy** may use                                 | Distance-to-goal, heading, height scan                                         |
| **Privileged**    | Info used in reward/training but not by the deployed policy | Sometimes used in “teacher” setups; we keep Phase A honest: policy sees o only |


**Why it matters:** On Mars you cannot give the rover a perfect map of every pebble. Training with only onboard-like o keeps the skill transferable.

### Observation space

The mathematical set of allowed observations — for us, a **fixed-length vector** (a `Box` in Gymnasium jargon):

 o∈R`n` 

which is read as:

> **"The observation vector o belongs to the n-dimensional real vector space."** 



For the rover project, an example might be:

o = [x ​y ​z ​vx ​​vy ​​vz ​​ϕ ​θ ​ψ​ d1​​⋯​dm​​]⊤∈Rn, 

where the observation vector contains the rover's position, velocity, orientation, and sensor measurements.



PPO’s neural net needs a **consistent size** n every step.

### Action space

The set of allowed actions. Ours is continuous and tiny:



a=[vω​]∈[vmin​,vmax​]×[ωmin​,ωmax​],

where:

- a is the action vector,
- v is the commanded linear velocity,
- ω is the commanded angular velocity,
- vmin⁡ and vmax​ are the allowable bounds on the linear velocity,
- ωmin⁡ and ωmax⁡​ are the allowable bounds on the angular velocity.

 

For a robotics or reinforcement learning paper, you could write:

a=[vω​]∈A,

where the action space is defined as

 A=[vmin​,vmax​]×[ωmin​,ωmax​].



| Symbol | Name             | Meaning                                 |
| ------ | ---------------- | --------------------------------------- |
| v      | Linear velocity  | Forward/back speed of the chassis (m/s) |
| \omega | Angular velocity | Yaw rate — how fast we turn (rad/s)     |


**Gymnasium** `Box`**:** A `Box` represents a **continuous action or observation space**, where each variable can take **any real value within specified lower and upper bounds**. Geometrically, this forms an nnn-dimensional rectangle (or hyperrectangle). For example, a rover's linear velocity may vary continuously between −1 and 1 m/s, while its angular velocity may vary between −2 and 2 rad/s. This contrasts with a `Discrete` space, where the agent selects from a finite set of predefined actions (e.g., "move left," "move right," or "stay still"), as commonly used in simple Q-learning environments like FrozenLake or Taxi. Continuous control tasks, such as robotics, autonomous driving, and drone navigation, almost always use a `Box` space because motors and actuators accept a continuous range of commands rather than a few fixed options.



### Policy πθ​(a∣o)

“Given what I see, sample (or pick) an action.”

  
Proximal Policy Optimization (PPO) learns the policy parameters θ of πθ​(a∣o). Chapter 4 defines only the observation and action spaces, o∈O⊆Rn and a∈A⊆Rm, whereas Chapter 7 optimizes θ.



### Manager-based API (Isaac Lab)

Instead of one giant env class, Lab uses **managers** that plug in **terms**:


| Manager             | Terms we care about now           |
| ------------------- | --------------------------------- |
| Observation manager | distance, heading, last_action, … |
| Action manager      | our differential-drive term       |


**Config classes** (`ObservationsCfg`, `ActionsCfg`) are the **wiring diagram**: which function, which scale, which sensor name.

**Industry parallel:** dependency injection / plugin architecture — swap a reward term without rewriting the whole simulator loop.

### Action term vs raw joint commands


| Layer         | Dimensionality                   | Who outputs it                 |
| ------------- | -------------------------------- | ------------------------------ |
| Policy action | 2: `[v, ω]`                      | Neural net (easy to learn)     |
| Action term   | Maps to 6 wheel velocity targets | Our code (robotics kinematics) |
| PhysX drive   | Tracks those targets with torque | Simulator                      |


This split is standard: **high-level command** vs **low-level actuator interface**.



## Differential (Skid-Steer) Kinematic Model

During **Phase A** of the project, the rover is modeled using a **differential (skid-steer) kinematic drive** rather than a full multi-body dynamic model. Although NASA's Perseverance rover possesses six independently driven wheels and four steering actuators, steering joints are intentionally omitted from the reinforcement learning action space in this initial implementation. Consequently, the RL agent controls only the rover's desired **linear velocity** and **angular velocity**,

a=[vω​],

where

- v is the commanded forward linear velocity (m/s),
- ω is the commanded yaw angular velocity (rad/s).

### Differential Drive Approximation

The commanded body velocities are converted into left- and right-side wheel velocities using the standard differential-drive kinematic equations:

vL​=v−B/2​ ω, 

vR​=v+B/2 ​ω,

where

- vL​ is the velocity commanded to the left wheels,
- vR is the velocity commanded to the right wheels,
- B is the effective track width (the distance between the centers of the left and right wheel sets).

These equations ensure that:

- when ω=0, both sides receive the same velocity and the rover moves in a straight line;
- when v=0, the wheels rotate in opposite directions, causing the rover to pivot about its center;
- when both v and ω are nonzero, the rover follows a curved trajectory.

### Wheel Angular Velocity

Isaac Lab controls wheel joints through **angular velocity** rather than linear velocity. Therefore, the commanded linear wheel velocities are converted into wheel rotational speeds using

ϕ˙​=vwheel / r​​​​,

where

- ϕ˙​ is the wheel angular velocity (rad/s),
- vwheel​ is the commanded linear velocity of the wheel,
- r is the wheel radius.

The six wheel joints are grouped into two synchronized sets:

ϕ˙​FL​=ϕ˙​ML​=ϕ˙​RL​ϕ˙​FR​=ϕ˙​MR​=ϕ˙​RR​​=rvL​​,=rvR​​,​ 



where

- FL = Front Left,
- ML = Middle Left,
- RL = Rear Left,
- FR = Front Right,
- MR = Middle Right,
- RR = Rear Right.

Thus, all three wheels on each side rotate at the same commanded speed.

### Engineering Rationale

This approach models the rover as a **skid-steered vehicle**, similar to a tracked robot or tank. Although it does not explicitly model the steering actuators, suspension geometry, or rocker-bogie mechanism of the real Perseverance rover, it captures the essential relationship between body motion and wheel velocities. This simplified representation significantly reduces the complexity of the control problem while remaining sufficient for learning obstacle avoidance, path planning, and autonomous navigation policies.

The objective of **Phase A** is therefore not to reproduce the full mechanical behavior of the Perseverance rover, but rather to enable efficient learning of navigation strategies in realistic Martian terrain. Once a robust navigation policy has been learned, more accurate vehicle dynamics—including steering joints, rocker-bogie suspension, wheel-terrain interaction, and slip effects—can be incorporated in later phases without fundamentally changing the reinforcement learning framework.



### Vectorization

Isaac Lab runs **many envs in parallel** on GPU. Every observation function returns a tensor shaped like `(num_envs, feature_dim)`.  
That’s why you see `torch` everywhere — industry-scale RL is batched.

---



## 4.2 Phase A observation design (what goes in o)

We intentionally **do not** feed raw cameras yet (Ch.8). Phase A vector:


| Term               | Approx. dim         | Intuition                  | Puzzle link                                |
| ------------------ | ------------------- | -------------------------- | ------------------------------------------ |
| `distance_to_goal` | 1                   | How far to the destination | Progress (Ch.5 will reward shrinking this) |
| `heading_to_goal`  | 1                   | Angle error in body frame  | “Am I pointed the right way?”              |
| `base_lin_vel`     | 3 (or 1 forward)    | How fast I’m moving        | Stops the policy flying blind              |
| `base_ang_vel`     | 3 (or 1 yaw)        | How fast I’m turning       | Same                                       |
| `last_action`      | 2                   | Previous `[v, ω]`          | Smoothness / Markov hint                   |
| `height_scan`      | H\times W flattened | Local terrain under/ahead  | Stand-in for HazCam awareness              |


## Goal-Relative Observations Instead of World Coordinates

Rather than providing the reinforcement learning agent with its **absolute position** in the world (e.g., Cartesian coordinates (x,y)(x,y)(x,y)), the observation space is designed using **goal-relative quantities**, such as the **distance to the goal** and the **heading error** between the rover's current orientation and the direction of the target.

Let the rover position be

p=[xy],\mathbf{p} = \begin{bmatrix} x\\ y \end{bmatrix},p=[xy​],

and the goal position be

g=[xgyg].\mathbf{g} = \begin{bmatrix} x_g\\ y_g \end{bmatrix}.g=[xg​yg​​].

The Euclidean distance to the goal is

d=∥g−p∥2,d = \|\mathbf{g}-\mathbf{p}\|_2,d=∥g−p∥2​,

while the desired heading toward the goal is

ψg=atan2⁡(yg−y,  xg−x).\psi_g = \operatorname{atan2}(y_g-y,\;x_g-x).ψg​=atan2(yg​−y,xg​−x).

If the rover's current heading is ψ\psiψ, then the heading error is

Δψ=wrap⁡(ψg−ψ),\Delta\psi = \operatorname{wrap}(\psi_g-\psi),Δψ=wrap(ψg​−ψ),

where the wrap operation confines the angle to the interval

[−π,π].[-\pi,\pi].[−π,π].

These relative quantities describe **how far the rover is from the goal and how much it must rotate to face it**, regardless of where the rover is located on the terrain.

Using goal-relative observations allows the policy

πθ(a∣o)\pi_{\theta}(a \mid o)πθ​(a∣o)

to learn a **general navigation strategy** rather than memorizing behaviors associated with specific regions of the map. For example, if the rover learns that a goal is five meters ahead and 30∘30^\circ30∘ to the left, it can apply the same steering behavior anywhere in the environment, whether it starts in the northwest corner or the southeast corner of the terrain.

In contrast, using absolute world coordinates (x,y)(x,y)(x,y) encourages the policy to associate particular actions with particular locations. Such a policy often performs well only on the training map and generalizes poorly to new terrains or different starting positions.

Consequently, representing the environment in a goal-relative reference frame improves **policy generalization**, enabling the trained agent to operate effectively across unseen environments and initial conditions.

---

## Observation Normalization and Feature Scaling

The observation vector supplied to the neural network contains quantities measured in different physical units, including distances (meters), angles (radians), velocities (m/s), and sensor readings. These variables often differ substantially in numerical magnitude.

For example,

- distance to the goal may range from

0≤d≤100 m,0 \le d \le 100 \text{ m},0≤d≤100 m,

while

- heading error lies within

−π≤Δψ≤π.-\pi \le \Delta\psi \le \pi.−π≤Δψ≤π.

If these values are supplied directly to the neural network, the larger numerical quantities can dominate the smaller ones during gradient-based optimization. As a result, the learning algorithm may place disproportionate emphasis on features with larger magnitudes, slowing convergence and reducing training stability.

To avoid this issue, observation variables are **normalized** so that most inputs have values on the order of one,

O(1).O(1).O(1).

A common normalization scheme is

d~=ddmax⁡,\tilde{d} = \frac{d}{d_{\max}},d~=dmax​d​,ψ~=Δψπ,\tilde{\psi} = \frac{\Delta\psi}{\pi},ψ~​=πΔψ​,v~=vvmax⁡,\tilde{v} = \frac{v}{v_{\max}},v~=vmax​v​,

where

- dmax⁡d_{\max}dmax​ is the maximum expected navigation distance,
- π\piπ normalizes angular measurements to the interval

[−1,1],[-1,1],[−1,1],

- vmax⁡v_{\max}vmax​ is the maximum allowable rover velocity.

The normalized observation vector becomes

o~=[d~ψ~v~⋮],\tilde{o} = \begin{bmatrix} \tilde{d}\\ \tilde{\psi}\\ \tilde{v}\\ \vdots \end{bmatrix},o~=​d~ψ~​v~⋮​​,

whose components have comparable numerical scales.

Maintaining observation values near unity improves the conditioning of the optimization problem, leading to **more stable gradients**, **faster convergence**, and **more reliable policy learning**. For this reason, observation normalization is considered a standard practice in modern reinforcement learning and is widely employed in Isaac Lab, Stable-Baselines3, and other deep reinforcement learning frameworks.

---



## 4.3 Phase A action design


| Design Aspect          | Chosen Implementation                                                                                                                                        | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Action Space Dimension | **2-dimensional action vector** a=[vω]                                                                                                                       | The reinforcement learning agent controls only the rover's desired **linear velocity** (vvv) and **angular velocity** (ω\omegaω). Although the rover has six independently driven wheels, directly controlling each wheel would require a six-dimensional (or higher) action space, making the learning problem significantly more difficult. A low-dimensional action space improves sample efficiency and allows the agent to learn navigation behaviors more quickly. |
| Action Range           | Linear and angular velocities are constrained to predefined bounds, v∈[vmin​,vmax​] and ω∈[ωmin⁡,ωmax⁡]                                                      | Clipping the actions prevents the policy from issuing physically unrealistic commands, such as excessive forward speeds or unrealistically rapid turns. These limits also improve training stability by ensuring that the rover always operates within safe and achievable operating conditions.                                                                                                                                                                         |
| Action Mapping         |                                                                                                                                                             | Isaac Lab actuates the rover through the **wheel revolute joints**, not by directly commanding body motion. The differential-drive mapping converts the agent's high-level navigation commands into individual wheel angular velocities that are compatible with the rover's drivetrain. This provides an intuitive control interface while remaining consistent with the robot's mechanical structure.                                                                  |
| Policy output scaling  | The neural network typically produces normalized actions in the interval [−1,1], which are subsequently scaled to the physical velocity limits of the rover. | Producing normalized outputs is standard practice in modern reinforcement learning because neural networks train more effectively when their outputs have a consistent numerical range. After inference, these normalized values are linearly mapped to the rover's allowable linear and angular velocity ranges before being executed by the simulator.                                                                                                                 |


**Path planning reminder:** Although the rover possesses **six independently driven wheels**, the reinforcement learning policy **does not learn six separate wheel commands**. Instead, it learns a high-level navigation command consisting of a desired forward velocity and turning rate. A deterministic kinematic controller then translates these commands into synchronized wheel velocities for the left and right wheel sets. This hierarchical control architecture reduces the complexity of the learning problem while still producing physically realistic rover motion in Isaac Lab.

---



## 4.4 Files we add (map)

```text
mars_rover/
├── mdp/
│   ├── kinematics.py          ← pure math (Mac-testable) twist → wheel speeds
│   ├── observations.py        ← obs term functions (need Isaac Lab at runtime)
│   └── actions/
│       ├── actions_cfg.py     ← ActionTermCfg dataclass
│       └── differential_drive.py  ← ActionTerm: process + apply
└── envs/navigation/
    ├── mdp/__init__.py        ← re-export terms for configs
    └── config/
        ├── observations_cfg.py
        └── actions_cfg.py
tests/
└── test_kinematics.py         ← run on Mac without Isaac
```

**Why pure** `kinematics.py`**?**  
Industry habit: keep **math** separable from **simulator bindings** so you can unit-test logic on a laptop without spinning Omniverse.

---



## 4.5 Code tour — `kinematics.py` (full)

**Path:** `source/mars_rover_rl/mars_rover/mdp/kinematics.py`

This file has **no Isaac dependency**. It is the “gearbox math.”

```python
"""Chassis twist [v, ω] → six wheel angular velocities (skid-steer model)."""

from __future__ import annotations

import torch

# Default order we will use everywhere (must match joint naming later).
WHEEL_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")


def twist_to_wheel_velocities(
    linear_x: torch.Tensor,
    angular_z: torch.Tensor,
    *,
    track_width: float,
    wheel_radius: float,
) -> torch.Tensor:
    """Convert chassis commands to wheel spin rates (rad/s).

    Args:
        linear_x: Forward speed v (m/s), shape (N,) or (N, 1).
        angular_z: Yaw rate ω (rad/s), same shape.
        track_width: Distance between left and right wheel centers (m), ``B``.
        wheel_radius: Wheel radius r (m).

    Returns:
        Tensor shape (N, 6) for wheels FL, FR, ML, MR, RL, RR in rad/s.
    """
    v = linear_x.squeeze(-1)
    w = angular_z.squeeze(-1)

    # Skid-steer / differential drive:
    #   v_L = v - ω * B/2
    #   v_R = v + ω * B/2
    half_b = track_width * 0.5
    v_left = v - w * half_b
    v_right = v + w * half_b

    # Linear wheel speed → angular rate: φ̇ = v_wheel / r
    w_left = v_left / wheel_radius
    w_right = v_right / wheel_radius

    # Three left wheels share w_left; three right share w_right.
    return torch.stack([w_left, w_right, w_left, w_right, w_left, w_right], dim=-1)
```


| Concept in code        | Plain English                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| `track_width` B        | How wide the rover is — larger B means the same \omega needs a bigger left/right speed difference |
| `wheel_radius` r       | Converts “meters per second at the rim” into “radians per second at the axle”                     |
| `torch.stack … dim=-1` | Build a feature dimension for 6 wheels per env                                                    |


**Try it (Mac):** after install, `pytest tests/test_kinematics.py -q` (see §4.9).

---



## 4.6 Code tour — observation terms

**Path:** `source/mars_rover_rl/mars_rover/mdp/observations.py`

These functions are what Isaac Lab’s observation manager **calls every step**.  
Signature pattern: `(env, **params) -> Tensor[num_envs, dim]`.

They **import Isaac Lab** — run them for real on the NVIDIA machine once the env exists. On Mac, read and understand; don’t expect `import isaaclab` to work unless you installed Lab there (you didn’t).

(See the file in the repo for the full commented source.)

**Ideas encoded:**


| Function           | Returns | Needs from env                        |
| ------------------ | ------- | ------------------------------------- |
| `distance_to_goal` | (N,1)   | Goal command / pose relative to robot |
| `heading_to_goal`  | (N,1)   | Same, using `atan2`                   |
| `last_action`      | (N,2)   | Action manager buffer                 |


Until Ch.6 wires a real goal command, some terms are **stubs** that return zeros but show the correct shapes — so configs can exist and you can see the plumbing.

---



## 4.7 Code tour — differential drive `ActionTerm`

**Paths:**

- `mars_rover/mdp/actions/actions_cfg.py` — config knobs  
- `mars_rover/mdp/actions/differential_drive.py` — runtime term

**Lifecycle each control step:**

1. PPO outputs `a` shape `(num_envs, 2)`
2. `process_actions`: scale/clip → store
3. `apply_actions`: kinematics → `set_joint_velocity_target` on the six joints

That is exactly the industry split: **controller command** → **inverse kinematics / mixing** → **actuator interface**.

---



## 4.8 Code tour — config wiring (`ObservationsCfg` / `ActionsCfg`)

**Paths under** `envs/navigation/config/`

These dataclasses don’t “run” physics. They **name the plugs**:

```text
ObsTerm(func=mdp.distance_to_goal, scale=0.1)
         │                    │
         │                    └─ how we shrink meters into friendlier numbers
         └─ function object the manager will call
```

When Ch.5–7 arrive, `NavigationEnvCfg` will include:

```text
observations = ObservationsCfg()
actions = ActionsCfg()
rewards = ...
```

Like a bill of materials for the MDP.

---



## 4.9 What you should run now (Mac)

```bash
cd ~/Projects/Mars_rover_rl
source .venv/bin/activate
python -m pip install torch pytest -q   # once (if missing)
python -m pytest tests/test_kinematics.py -q
```

**Pass:** kinematics tests green.  
That proves the action *math* puzzle piece without Isaac.

### If you see `No module named 'torch'`

Your prompt may show `(.venv) (base)` — **conda base** is still active and `pytest` may be Anaconda’s, not the venv’s.

Check:

```bash
which python
which pytest
python -c "import sys; print(sys.executable)"
```

You want paths under `.../Mars_rover_rl/.venv/...`.  
If you see `/opt/anaconda3/...`, either:

```bash
conda deactivate    # repeat until (base) is gone
source .venv/bin/activate
python -m pip install torch pytest
python -m pytest tests/test_kinematics.py -q
```

or always force the venv interpreter (bulletproof):

```bash
cd ~/Projects/Mars_rover_rl
.venv/bin/python -m pip install torch pytest
.venv/bin/python -m pytest tests/test_kinematics.py -q
```

Using `python -m pytest` (not bare `pytest`) guarantees the same Python that has `torch`.

**Do not use** `pytest -m tests/...` — in pytest, `-m` means “filter by mark,” not “module path.” That yields `4 deselected` and runs nothing. The path is a normal argument:

```bash
python -m pytest tests/test_kinematics.py -q    # correct
# pytest -m tests/test_kinematics.py            # wrong: -m is mark filter
```

If `(base)` stays in the prompt after `conda deactivate`, run `conda deactivate` again until it disappears (nested conda activations are common). `which python` pointing at `.venv` is what actually matters.

---



## 4.10 How this connects to PPO (preview of Ch.7)

```text
o_t  --πθ-->  a_t=[v,ω]  --action term-->  wheel targets  --PhysX-->  new o_{t+1}, r_t
```

PPO updates \theta so actions that led to high **advantage** become more likely.  
If observations are garbage or actions can’t express needed motions, PPO cannot save you — **spaces are part of algorithm design**, not just plumbing.

---



## 4.11 RLRoverLab inspiration (puzzle, not photocopy)


| They do                             | We do                                                      |
| ----------------------------------- | ---------------------------------------------------------- |
| Obs: distance, heading, height scan | Same Phase A philosophy                                    |
| Action dim 2: lin + ang             | Same                                                       |
| Ackermann steering joints           | **Skid-steer / 6 wheel speeds** (matches our articulation) |
| Lab 2.x + skrl                      | Lab 3 + RSL-RL later                                       |


---



## 4.12 Checklist

- [ ] I can explain state vs observation with a Perseverance example  
- [ ] I can explain why action dim is 2, not 6  
- [ ] I can write the skid-steer formulas for v_L, v_R from memory  
- [ ] I know which file is Mac-testable vs Isaac-only  
- [ ] `pytest tests/test_kinematics.py` passes  
- [ ] I read `differential_drive.py` comments once end-to-end  

---



## 4.13 Next chapter

**Chapter 5 — Rewards & terminations:** turn “drive safely to the goal” into numbers r_t and done flags — the teaching signal PPO actually optimizes.

---



## 4.14 North star

**Observations are the rover’s senses; actions are its intents; the action term is the gearbox; PPO will learn which intents are wise — but only after rewards (Ch.5) and a Mars world (Ch.6) exist.**