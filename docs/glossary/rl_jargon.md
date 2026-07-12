# RL & robotics jargon (living sheet)

Short definitions. Expanded explanations live in the chapters that introduce them.

| Term | One-liner |
|------|-----------|
| **MDP** | Markov Decision Process — the formal game (obs/actions/rewards/transitions). |
| **State \(s\)** | Full sim truth (may include privileged info). |
| **Observation \(o\)** | What the policy is allowed to see. |
| **Action \(a\)** | Command issued by the policy (here: \([v,\omega]\)). |
| **Policy \(\pi\)** | Mapping from observation to action distribution (the “driver”). |
| **Episode** | One attempt: reset → steps → termination. |
| **Return** | Discounted sum of future rewards. |
| **Value \(V\)** | Expected return from an observation (critic’s job). |
| **Advantage \(A\)** | How much better an action was than expected. |
| **On-policy** | Learn only from data the current policy collected (PPO). |
| **PPO** | Proximal Policy Optimization — clipped policy-gradient method. |
| **Actor–Critic** | Two nets: policy (actor) + value (critic). |
| **Rollout** | Batch of experience gathered before a PPO update. |
| **RSL-RL** | GPU RL library commonly paired with Isaac Lab for PPO. |
| **Manager-based env** | Isaac Lab style: obs/reward/action/termination as pluggable terms. |
| **Privileged info** | Signals available in sim (e.g. exact rock positions) not on the real rover. |
| **Domain randomization** | Randomize sim parameters so policies transfer better. |
| **Reach-avoid** | Formal goal: reach a target set while avoiding an unsafe set. |
| **Height scan** | Ray-cast elevation grid used as a compact terrain observation. |
| **Rocker-bogie** | Passive suspension that keeps wheels on uneven ground. |
| **Empty (Blender)** | Pose-only object (no mesh); we use them as camera/joint markers. |
| **Keep Transform** | Parent without moving the object in world space. |
| **Origin** | Object-local (0,0,0); wheel origins should sit at the hub/axle. |
| **USD** | Scene description format Isaac reads (meshes, hierarchy, later physics). |
| **Prim** | One node in a USD stage (like one object in an Outliner). |
| **Xform** | Prim that stores a pose; Blender Empties often become Xforms. |
| **Isaac Sim** | Interactive simulator/editor where you debug physics visually. |
| **Isaac Lab** | Python RL framework on top of Isaac Sim (envs, training). |
| **PhysX** | Physics engine computing contacts, joints, gravity. |
| **Rigid body** | A solid PhysX can move (has mass). |
| **Revolute joint** | Hinge joint — one rotational degree of freedom (wheel axle). |
| **Drive** | Motor on a joint; we use velocity targets for wheels. |
| **Collider** | Simplified collision shape (box/cylinder), separate from visuals. |
| **Articulation** | Multi-link robot with joints, solved as one PhysX system. |
| **ArticulationCfg** | Isaac Lab config that spawns/controls that articulation in code. |
| **DOF** | Degree of freedom — an independent axis of motion. |

Add rows as new chapters introduce terms.
