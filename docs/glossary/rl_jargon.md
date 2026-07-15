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
| **Observation space** | Set of allowed observations (ours: a fixed-length vector). |
| **Action space** | Set of allowed actions (ours: continuous `[v, ω]`). |
| **Box space** | Gymnasium continuous range (vs Discrete button choices). |
| **Policy π** | Neural mapping from observation → action distribution. |
| **Action term** | Lab plugin that turns policy actions into joint targets. |
| **Observation term** | Lab plugin function that extracts one piece of `o_t`. |
| **Manager-based env** | Lab style: MDP built from pluggable terms + configs. |
| **Privileged info** | Sim-only signals not given to the deployed policy. |
| **Skid-steer / differential drive** | Left/right wheel speed mixing from chassis `[v, ω]`. |
| **Vectorization** | Running many envs in parallel as batched tensors. |
| **Generalization** | Doing well on new goals/terrains, not memorizing one map. |
| **Editable install (`pip -e`)** | Link a package into an env so code edits apply immediately. |
| **Reward \(r_t\)** | Scalar grade the env returns after each action. |
| **Return \(R_t\)** | Discounted sum of future rewards from time t. |
| **Discount \(\gamma\)** | How much the agent values future vs immediate reward (often ~0.99). |
| **Reward shaping** | Adding intermediate rewards so learning is easier (can be exploited). |
| **Reward hacking** | Policy finds loopholes that maximize reward without doing the real task. |
| **Dense / sparse reward** | Feedback every step vs only on rare events (goal/crash). |
| **Termination** | Episode should end now for a task reason (success, crash, …). |
| **Truncation** | Episode cut by time limit (not necessarily a semantic failure). |
| **Reward term** | One ingredient function in a weighted multi-term reward. |
| **Terrain** | Driveable ground surface (height field or mesh). |
| **Procedural generation** | Building worlds from parameters/algorithms, not one hand-made level. |
| **Obstacle** | Something the rover should not drive through (rocks, pillars, …). |
| **Goal / target** | Destination the rover should reach this episode. |
| **Command (Isaac Lab)** | Per-env target signal (e.g. `target_pose`) read by obs/rewards. |
| **Reachable goal** | A destination that is possible without cheating through geometry. |
| **Scene cfg** | Code shopping list for one env: ground, robot, lights, sensors. |
| **Event / reset** | Lifecycle hook that repositions robot / randomizes on episode start. |
| **Env spacing** | Distance between parallel cloned environments in the world. |
| **Curriculum** | Gradually increasing task difficulty over training. |
| **Domain randomization** | Randomizing sim details so policies generalize better. |
| **Gymnasium / gym.make** | Standard API to construct envs by string task id. |
| **gym.register** | Advertise a task id + entry points so `gym.make` can find it. |
| **Entry point** | String address of a Python class (`module:Class`). |
| **Wrapper** | Adapter between env API and an RL library’s expected interface. |
| **RSL-RL** | High-performance robotics RL library (PPO runner used with Isaac Lab). |
| **OnPolicyRunner** | RSL-RL object that collects rollouts and runs PPO updates. |
| **Rollout** | Batch of experience collected before a policy update. |
| **Iteration (training)** | One collect-experience + learn cycle. |
| **Checkpoint** | Saved network weights you can reload for resume/eval. |
| **Hyperparameter** | Training knob you set (LR, gamma, clip…) not learned by SGD. |
| **Decimation** | Physics substeps per one RL action. |
| **Headless** | Run sim without a live GUI viewport, to skip rendering cost; metrics still log normally. |
| **TensorBoard** | Tool to plot training curves from log directories. |
| **Live viewport** | Isaac Sim's real-time 3D window — good for quick sanity checks, too slow for full training. |
| **Video recording (`--video`)** | Periodic short `.mp4` clips saved during a headless run, for reviewing behavior later without a live window. |
| **TiledCamera** | Isaac Lab's render-based camera sensor built for RL — batches all parallel envs into one tiled render pass. |
| **Depth image** | Per-pixel distance-to-camera; easier to learn navigation from than raw RGB. |
| **Observation group** | A named bundle of obs terms (e.g. `policy`, `vision`); can be concatenated into one vector or kept as separate tensors. |
| **`concatenate_terms`** | Obs-group flag: `True` stacks terms into one flat vector (MLP-friendly); `False` keeps shapes intact (needed for images). |
| **CNN encoder** | Small conv-net that compresses an image observation into a fixed-size feature vector before fusing with other obs. |
| **Feature fusion** | Concatenating features from different modalities (e.g. CNN image features + vector obs) before the actor/critic heads. |
| **Asymmetric actor-critic** | Critic sees more (privileged/extra) info than the actor, which only sees what a deployed policy would have onboard. |
| **Reach set \(\mathcal{G}\) / avoid set \(\mathcal{U}\)** | Formal reach-avoid targets: states you want to reach vs. states you must never enter. |
| **HJ reachability** | Hamilton-Jacobi PDE approach to exactly compute reach-avoid guarantees; research-grade, not implemented here. |
| **Safety filter / shield** | Runtime check that overrides a policy's action when it looks unsafe; lightweight stand-in for full HJ reachability. |
| **Control barrier function (CBF)** | Scalar function whose sign indicates safe vs. unsafe; formal basis for principled safety filters. |
| **Backup policy** | The fallback action (e.g. "stop") a shield substitutes for an unsafe policy action. |

Add rows as new chapters introduce terms.
