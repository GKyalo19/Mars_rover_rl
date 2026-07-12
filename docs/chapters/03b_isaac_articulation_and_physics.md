# Chapter 3b — Isaac Sim / Lab: Articulation & Physics (beginner guide)

> **Prerequisite:** Blender cleanup from [Chapter 3](03_rover_asset_pipeline.md) (named wheels, `base_link`, camera Empties, USD export).  
> **Where you run this:** NVIDIA machine with **Isaac Sim 6.0** + **Isaac Lab 3.0**.  
> **Goal:** Turn a visual USD into a *drivable* robot Isaac Lab can train with PPO.  
> **Style:** Each tool is explained as **what it is → why we need it → how it fits Perseverance**.

You do not need to memorize every Omniverse menu. You need the *ideas*. UI labels can shift slightly between Sim versions; the concepts stay stable.

---

## 3b.1 The cast of characters (what is what)

People say “Isaac” loosely. These are different layers:


| Name                | What it is                                                                            | Role in *this* project                                                   |
| ------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **USD**             | Universal Scene Description — a scene file format (like a powerful 3D “project file”) | Your rover mesh hierarchy lives here after Blender export                |
| **Prim**            | One node in the USD stage (`/World/Rover/wheel_FL`)                                   | Every body, joint, camera is a prim                                      |
| **Xform**           | A prim that mainly stores a transform (pose)                                          | Your Blender **Empty** usually becomes an Xform — a named frame in space |
| **Isaac Sim**       | The interactive simulator / editor (GUI + PhysX)                                      | Where you open the USD, click, add joints, press Play, watch wheels spin |
| **PhysX**           | The physics engine under Isaac                                                        | Computes contacts, gravity, joint constraints every sim step             |
| **Isaac Lab**       | Python framework *on top of* Isaac Sim for robot learning                             | Defines envs, observations, rewards, vectorized training                 |
| **Articulation**    | A robot made of links connected by joints, solved as one system                       | Perseverance chassis + 6 wheels (v1)                                     |
| **ArticulationCfg** | Isaac Lab config class describing that robot for code                                 | “Here is the USD path, here are the joints, here is the initial pose”    |
| **RSL-RL**          | PPO training library Lab talks to                                                     | Does not care about meshes — only tensors from the env                   |


**Analogy:**

- **USD** = sheet music and stage layout  
- **Isaac Sim** = the theater where you rehearse with physics  
- **Isaac Lab** = the coach running hundreds of rehearsals in parallel for RL  
- **PPO / RSL-RL** = the student learning the driving policy

---

## 3b.2 Big picture: what we are building in Isaac

```text
Visual USD from Blender
        │
        ▼
┌───────────────────────────────┐
│  Articulation (physics robot) │
│  • base_link  (rigid body)    │
│  • wheel_*    (rigid bodies)  │
│  • 6 revolute joints + drives │
│  • collision shapes           │
│  • camera prims at markers    │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Isaac Lab ArticulationCfg    │
│  + Navigation env (later ch.) │
│  Action: [v, ω] → wheel ωdot  │
└───────────────────────────────┘
```

**Phase A simplification (intentional):**  
Wheels are hinged directly to `base_link`. The pretty `suspension` mesh can be **visual only** (no extra joints). Real rocker-bogie can come later. This matches Chapter 1: learn navigation first.

---

## 3b.3 Core physics ideas (learn these once)

### Rigid body

**What:** A solid chunk PhysX can move. It has mass, center of mass, inertia, and (usually) a collision shape.

**Why:** Without rigid bodies, meshes are decorations — gravity and wheels do nothing.

**For us:** `base_link` (chassis) + each `wheel_FL` … `wheel_RR` become rigid bodies. The decorative suspension mesh can be *kinematic/visual* or welded to the chassis for v1.

### Link

**What:** One rigid piece of an articulation (robotics word for “body segment”).

**For us:** 7 links in v1 → chassis + 6 wheels.

### Joint

**What:** A constraint that says how two links may move relative to each other.

**Revolute joint:** Rotation about **one axis** only (like a hinge). Perfect for a wheel spinning on an axle.

**Why not “just move the mesh”?** If you teleport wheel rotation in code without a joint, contacts and rolling physics are wrong. Joints let PhysX enforce “this wheel can spin, but cannot fall off the axle.”

**For us:** Six revolute joints: `base_link` → each `wheel_`*, axis = axle direction (usually the wheel’s lateral axis).

### Drive (on a joint)

**What:** A motor attached to a joint. You command a target (often **velocity** or position) and PhysX applies torque to track it.

**Why:** PPO should not output raw torques at first. It outputs chassis twist `[v, ω]`; our action term converts that to **six wheel angular velocity targets**; drives try to achieve those speeds.

**For us:** Velocity drives on each wheel revolute joint.

### Collision shape (collider) vs visual mesh

**What:**

- **Visual mesh** = what you see (detailed Perseverance art, 90k+ faces).  
- **Collision shape** = simplified volume PhysX uses for contact (cylinder, box, convex hull).

**Why:** Simulating collisions on every artistic triangle is slow and jittery. RL needs *stable, fast* contacts.

**For us:**


| Part          | Typical collider                                       |
| ------------- | ------------------------------------------------------ |
| Wheel         | **Cylinder** (or thin capsule) aligned with axle       |
| Chassis       | **Box** or a few boxes                                 |
| Rocks/terrain | Terrain heightfield / mesh collider on the ground side |


Visual suspension can have **no collider** at first (or a coarse box) so it does not fight the wheel colliders.

### Contact / friction

**What:** When colliders touch, PhysX applies normal forces and friction.

**Why the rover drives:** Wheel cylinder pushes into terrain → friction turns spin into forward motion. Bad friction or intersecting colliders at start = exploding rover (classic Isaac rite of passage).

### Articulation (again, practically)

**What:** PhysX treats the whole robot as one articulated system (more stable than many separate rigid bodies glued with generic constraints).

**Why Isaac Lab likes it:** Sensors, joint readings, and resets are designed around articulations.

---

## 3b.4 Cameras in Isaac (how Empties pay off)


| Blender              | Isaac                            | Purpose                     |
| -------------------- | -------------------------------- | --------------------------- |
| Empty `cam_nav_left` | Xform / Camera prim at same pose | Place the sensor            |
| (no image yet)       | Camera prim with resolution, FOV | Actually render pixels      |
| —                    | Lab camera sensor wrapper        | Feed observations (Phase B) |


**What a Camera prim is:** A sensor that renders an image from a pose each step (or on demand).

**Why we did not need Blender cameras:** Blender’s camera object is for Blender rendering. Isaac has its own camera schema (resolution, clipping, projection). The Empty only needed to carry **where** and roughly **which way**.

**Phase A reminder:** Training can use a **height scanner** (ray casts) instead of cameras first. Still create camera prims when you can — they validate your markers and unlock Phase B.

---

## 3b.5 Isaac Sim GUI map (where things live)

When Isaac Sim 6.0 opens, learn these three panels (same idea as Blender’s Outliner / Viewport / Properties):


| Panel        | Usually where           | Like in Blender | You use it to…                                                         |
| ------------ | ----------------------- | --------------- | ---------------------------------------------------------------------- |
| **Stage**    | Left                    | Outliner        | Click prim names (`base_link`, `wheel_FL`, …)                          |
| **Viewport** | Center                  | 3D view         | See the rover; **Play** / **Stop** at the left of the viewport toolbar |
| **Property** | Bottom-right (or right) | Properties      | Click `**+ Add`**, edit mass, drive, joint axis                        |


Also useful:


| Control                     | Where                 | What it does                                       |
| --------------------------- | --------------------- | -------------------------------------------------- |
| **Play**                    | Viewport toolbar (▶)  | Starts PhysX — gravity, joints, drives become live |
| **Stop**                    | Same toolbar          | Resets / stops sim (stop before editing joints)    |
| **Eye icon** (viewport top) | Viewport overlay menu | Show collision outlines, etc.                      |
| **File → Open / Save As**   | Menu bar              | Load your USD / save articulated version           |


**Selection tips (Stage tree):**

- Click one prim to select it.  
- **Ctrl+click** to add more to the selection.  
- **Shift+click** for a consecutive range in the tree.  
- Selected prims highlight in the viewport.

**Official NVIDIA twin tutorials** (same clicks we use below):  
[Assemble a Simple Robot](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/tutorial_intro_assemble_robot.html) · [Articulate a Basic Robot](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/tutorial_gui_simple_robot.html)

---

## 3b.6 Click-by-click: turn Perseverance into a drivable robot

Do this on the **NVIDIA machine** after your Blender USD exists.  
**Stop the simulation** (not playing) before adding APIs or joints.

### Step 0 — Open the USD and add a ground (so it has somewhere to land)

1. Launch **Isaac Sim 6.0**.
2. Menu bar → **File → Open…** → choose `perseverance_visual.usd`
  (or **File → New**, then **File → Add Reference** if you prefer referencing — for first-time editing, **Open** the file you will modify is simpler).
3. In the **Stage** panel, expand the tree until you see `base_link` and `wheel_FL` … `wheel_RR`.
4. Add a floor so the rover does not fall forever:
  - Menu bar → **Create → Physics → Ground Plane**  
  - (If you do not see that exact path: **Create → Shape** for a large thin Cube, then give *that* a static collider later — Ground Plane is easier.)
5. Select your rover root (`base_link` or parent Xform). In **Property → Transform → Translate**, set **Z** high enough that the wheels sit **slightly above** the ground (e.g. start at `Z = 1.0` and lower later).

**Why:** If wheels spawn *inside* the ground collider, PhysX “explodes” the robot on Play.

**Checkpoint:** Press **Play** once with *no* rigid bodies yet — nothing should fall (meshes are still visual-only). Press **Stop**.

---

### Step 1 — Add Rigid Body + Colliders (the big “make it physical” click)

This is the step that used to say only “add rigid body.” Here are the exact clicks from Isaac Sim 6.0’s own tutorial.

1. In the **Stage** tree, select the **visual meshes** (or Xforms) for:
  - chassis / `base_link` body mesh  
  - `wheel_FL`, `wheel_FR`, `wheel_ML`, `wheel_MR`, `wheel_RL`, `wheel_RR`  
   Use **Ctrl+click** to multi-select all of them.  
   *(If each wheel is an Xform with a child mesh, select the mesh that has geometry — or the Xform if that is what you parented joints to later. Be consistent.)*
2. Look at the **Property** panel (with those prims still selected).
3. Near the top of Property, click the blue/gray button `**+ Add**`.
4. In the popup menu choose:
  **Physics → Rigid Body with Colliders Preset**
5. Scroll down in Property. You should now see new sections named something like:
  - **Rigid Body**
  - **Collider** (or Collision)

**What that preset did:**


| API it added   | Meaning                                                |
| -------------- | ------------------------------------------------------ |
| **Rigid Body** | This prim has mass and feels gravity / forces          |
| **Collider**   | PhysX can bump into a simplified shape around the mesh |


You *can* add them separately later via `**+ Add` → Physics → Rigid Body** and `**+ Add` → Physics → Collider** / **Collider Preset**, but the combined preset is the right first move.

1. **Optional mass tweak** (still in Property → Rigid Body):
  - Chassis: try something like `50`–`200` for a light training model (real Perseverance is ~1025 kg — fine later).  
  - Each wheel: smaller, e.g. `5`–`20`.  
   Exact numbers matter less than “chassis heavier than one wheel.”
2. Press **Play**.
  - Expect: pieces fall onto the ground plane.  
  - If they fall **separately** and scatter — good for now (no joints yet).  
  - If they **explode** — Stop, raise the rover in Z, or check overlapping colliders (next step).
3. Press **Stop**.

---

### Step 2 — See the collision shapes (purple outlines)

1. In the **viewport**, click the **eye** icon (Show / visibility menu at the top of the viewport).
2. Enable: **Show By Type → Physics → Colliders → All**.
3. You should see **purple** outlines around body and wheels.

**What you are looking at:** The *collision* mesh PhysX uses — not every artistic triangle.

1. Select one **wheel** mesh → Property → find **Collider** / **Collision** section → look for **Approximation** (wording may be “Collider Approximation”).
  - Default is often **Convex Hull** — OK to start.  
  - For wheels, **Convex Hull** of a cylinder-like wheel is usually fine.  
  - Avoid triangle mesh colliders on *dynamic* (moving) bodies — Isaac restricts those to static objects.
2. Select the **body** mesh → same check. If the purple shape is a huge blob that swallows the wheels, you may need a simpler approximation later (or separate visual vs collision prims). For v1, Convex Hull is acceptable if Play is stable.

**Do not** give detailed science instruments their own heavy colliders yet — hide or skip collision on junk if it causes jitter (`+ Add` was only on chassis + wheels).

---

### Step 3 — Add friction (so spinning wheels can push the ground)

Without friction, wheels spin and the rover may skate.

1. Menu bar → **Create → Physics → Physics Material**.
2. In the dialog, choose **Rigid Body Material**.
3. A new prim appears in the Stage (e.g. `PhysicsMaterial`). Rename it to `rover_rubber` (right-click → **Rename**).
4. Select `rover_rubber` → Property → set roughly:
  - **Static Friction** ≈ `1.0`  
  - **Dynamic Friction** ≈ `1.0`  
   (Tune later; these are starting values.)
5. Select all six **wheel** meshes (Ctrl+click).
6. In Property, find **Materials on selected models** (or “Physics materials” assignment — same idea as the assemble tutorial).
7. Choose `rover_rubber` from the dropdown.

**What you did:** Told PhysX “wheel–ground contact should grip,” which is how rotation becomes driving.

---

### Step 4 — Create revolute joints (hinge each wheel to the body)

Joints are **not** under `+ Add`. You create them from a **right-click menu** after selecting **two different rigid bodies**.

#### 4a — Fix hierarchy first (this prevents “joint between a body and itself”)

**Do not** joint wheel → `suspension` for v1. Suspension is visual only. Joint **chassis ↔ wheel**.

That error almost always means PhysX thinks both picks are the **same** rigid body. Common cause: wheels are **children** of `base_link`, and nested rigid bodies collapse into one body.

**Target Stage tree:**

```text
perseverance              ← Articulation Root goes HERE later (no Rigid Body on this)
├── base_link             ← Rigid Body (chassis)
│   ├── Body / Body_Parts / suspension   ← visuals only, NO Rigid Body of their own
├── wheel_FL              ← Rigid Body (sibling of base_link, NOT child)
├── wheel_FR
├── …
└── Joints/
    ├── wheel_joint_FL
    └── …
```

**Fix in Isaac (if wheels are currently under `base_link`):**

1. **Stop** simulation.
2. Right-click in Stage → **Create → Xform** → rename to `perseverance`.
3. Drag `base_link` under `perseverance`.
4. Drag each `wheel_*` **out** from under `base_link` and drop them under `perseverance` (same level as `base_link`).
5. Confirm each `wheel_*` and `base_link` still show their own **Rigid Body** section in Property.
6. Confirm `suspension` (if any) is under `base_link` and has **no** Rigid Body (if it has one, click the **X** on that Rigid Body section to remove it).

**Check:** Select `wheel_FL` alone → Property has Rigid Body. Select `base_link` alone → Property has Rigid Body. Two separate prims, two separate rigid bodies.

#### 4b — Optional: a folder for joints

1. Right-click under `perseverance` → **Create → Scope**.
2. Rename the Scope to `Joints`.
   *(A Scope is just a folder in the USD tree — organization only.)*

#### 4c — One wheel at a time (example: front left)

1. **Stop** if Play is running.
2. In Stage, click **`base_link`** (chassis rigid body).
3. Hold **Ctrl** and click **`wheel_FL`** (must be a **sibling** of `base_link`, not a child).
   Both should be highlighted — **two different prims**.
4. **Right-click** on the selection → **Create → Physics → Joints → Revolute Joint**
5. A new prim appears (often under the wheel), named like `RevoluteJoint`.
6. **Rename** it to `wheel_joint_FL`.
7. Select `wheel_joint_FL` → Property:
   - **Body 0** = chassis path (`…/base_link`)
   - **Body 1** = wheel path (`…/wheel_FL`)
   - If Body 0 and Body 1 are the **same path**, hierarchy is still wrong — go back to 4a.
8. Set **Axis** to the axle direction (try X / Y / Z until the wheel spins like a tire).
9. Drag `wheel_joint_FL` into the `Joints` Scope (optional).
10. Repeat for the other five wheels.

**Checkpoint without motors:** Press **Play**, hold **Shift**, click-drag the body. Wheels stay attached and can roll. Press **Stop**.

**What you did:** Mechanically attached wheels so they can spin but not fall off.

---

### Step 5 — Add Angular Drive (the motor) on each joint

1. In Stage, select all six `wheel_joint_*` prims (Ctrl+click).
2. Property panel → click `**+ Add**`.
3. Choose **Physics → Angular Drive**.
4. With the joints still selected (or edit one, then copy values), open the **Drive** / **Angular Drive** section in Property and set:


| Property            | Starting value         | Why                                               |
| ------------------- | ---------------------- | ------------------------------------------------- |
| **Stiffness**       | `0`                    | Velocity control (not “hold this angle”)          |
| **Damping**         | `10000` (`1e4`)        | Strong velocity tracking (NVIDIA’s wheel example) |
| **Target Velocity** | try `30`–`200` to test | Commanded spin rate                               |


**Units note (important):**  
In the **Property panel on the joint**, angular drive targets are often shown in **degrees/sec**.  
Isaac Lab / articulation controllers usually speak **radians/sec**. When we code later we will be careful; for this GUI test, just pick a number that makes the rover creep, not launch.

1. Press **Play**.
  - All six wheels should spin.  
  - With friction + ground, the rover should start **rolling**.
2. If it drives **backward**, set Target Velocity negative, or flip joint axis later.
3. Press **Stop**. Set Target Velocity back to `0` when you are done testing (so it does not always drive on open).

**What you did:** Added the actuators PPO’s action mapper will eventually command.

---

### Step 6 — Add Articulation Root (make it one robot, not loose parts)

1. Select the **rover root** prim in Stage — the top Xform **`perseverance`** (parent of `base_link` + wheels). Do **not** put Articulation Root only on a nested wheel.
2. Property → **`+ Add`**.
3. Choose **Physics → Articulation Root**.

**What this does:** Tells PhysX to solve chassis + wheels as one **articulation** (more stable, what Isaac Lab expects).

1. Press **Play** again — driving should still work, often more stably. **Stop**.

---

### Step 7 — (Optional now) Add a Camera on an Empty / Xform marker

1. In Stage, find a marker from Blender, e.g. `cam_nav_left` (an Xform).
2. Right-click that prim → **Create → Camera**
  (or menu **Create → Camera**, then parent it under the marker and zero its local translate).
3. Select the Camera → Property → set a modest resolution if shown.
4. In the viewport camera menu (often a camera icon / “Cameras”), choose your new camera to preview.
5. If you stare into the chassis, select the Camera → rotate **Orient** 180° about an axis until the view looks outward.

**Skip if you want:** Phase A training can use height scans; cameras are for Phase B. Markers make this step fast when you return.

---

### Step 8 — Save the articulated USD

1. **File → Save As…**
2. Name it e.g. `perseverance_articulated.usd` next to your visual USD.
3. Keep the visual-only file untouched as a backup.

---

## 3b.6.1 Click cheat-sheet (print this)


| Goal                  | Clicks                                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| Rigid body + collider | Select meshes → Property `**+ Add**` → **Physics → Rigid Body with Colliders Preset**                       |
| See colliders         | Viewport **eye** → **Show By Type → Physics → Colliders → All**                                             |
| Friction material     | **Create → Physics → Physics Material** → **Rigid Body Material** → assign on wheels                        |
| Wheel hinge           | Select **chassis** then **Ctrl+click wheel** → right-click → **Create → Physics → Joints → Revolute Joint** |
| Motor                 | Select joint(s) → `**+ Add**` → **Physics → Angular Drive** → Stiffness `0`, Damping `1e4`                  |
| Articulation          | Select rover root → `**+ Add**` → **Physics → Articulation Root**                                           |
| Ground                | **Create → Physics → Ground Plane**                                                                         |
| Test                  | Viewport **Play** / **Stop**                                                                                |


---

## 3b.6.2 If your menus look slightly different

Isaac Sim versions shuffle labels rarely, but if you cannot find an item:

1. With the prim selected, open Property `**+ Add**` and search the list for **Rigid Body**, **Collider**, **Angular Drive**, **Articulation Root**.
2. For joints, remember: they are under **right-click → Create → Physics → Joints**, not under `+ Add`.
3. Use the NVIDIA sample to practice the same clicks on a toy car first:
  Content Browser → `Isaac Sim/Samples/Rigging/MockRobot/`  
   That builds muscle memory before Perseverance’s denser mesh.

---

## 3b.7 How Isaac Lab wraps the same robot (`ArticulationCfg`)

Once the robot works under Play in Sim, Lab needs a **Python description** so training envs can spawn hundreds of copies.

Conceptually (names illustrative — we will match Lab 3.0 APIs when coding):

```text
ArticulationCfg
├── spawn: path to perseverance_articulated.usd (or USD + modifiers)
├── actuators: which joint names get velocity/effort limits
├── init_state: default joint positions / root pose
└── soft limits, collision props, etc.
```

**What ArticulationCfg does for the project:**


| Piece      | Meaning                                             |
| ---------- | --------------------------------------------------- |
| Spawn      | “Clone this robot into each parallel env”           |
| Actuators  | “These joint drives are controllable from code”     |
| Init state | “On reset, put rover here with wheels at zero spin” |


Then a **navigation env cfg** (later chapter) adds:

- terrain + goal command  
- observation terms  
- reward terms  
- action term: `[v, ω]` → six wheel velocity targets  
- terminations

**You are not behind if GUI comes before code.** GUI builds intuition; `ArticulationCfg` makes it reproducible.

---

## 3b.8 How actions flow (connecting to PPO)

```text
PPO actor outputs a_t = [v, ω]
        │
        ▼
Action term (our code)
  maps twist → [ω_FL, ω_FR, ω_ML, ω_MR, ω_RL, ω_RR]
        │
        ▼
Joint velocity drives (PhysX)
        │
        ▼
Wheels spin → friction → rover moves
        │
        ▼
Sensors / state → observation vector → next PPO step
```

**Jargon:**

- **Action space** = what PPO outputs (`Box(2,)` for `[v, ω]`).  
- **Actuators** = what the simulator motors accept (6 wheel speeds).  
- The mapper in between is *not* the neural net — it is robotics kinematics (differential-drive style mixing).

---

## 3b.9 Isaac Lab vs “raw” Isaac Sim (when to use which)


| Task                                                     | Prefer              |
| -------------------------------------------------------- | ------------------- |
| First-time joint debugging, “why is my wheel exploding?” | **Isaac Sim GUI**   |
| Defining obs/rewards/PPO training                        | **Isaac Lab**       |
| 1000 parallel Mars envs                                  | **Isaac Lab** + GPU |
| One-off screenshot of NavCam                             | Either              |


**Mental model:** Sim is the instrument; Lab is the experimental protocol.

---

## 3b.10 Common first-time failures (and what they teach)


| Symptom                       | Likely meaning                                      | What to learn                                       |
| ----------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| Rover explodes on Play        | Colliders overlapping terrain/each other at t=0     | Contacts need non-penetration initially             |
| Wheels spin but no drive      | No friction or collider not touching ground         | Motion comes from contact, not from “animation”     |
| Wheels leave the body         | Joint pivot/axis wrong or not an articulation joint | Joints define allowed motion                        |
| Flips immediately             | COM too high / tiny wheel colliders / huge torque   | Mass distribution & drive gains matter              |
| Camera looks into the chassis | Local −Z convention / Empty orientation             | Sensor frames need verification                     |
| “Joint between a body and itself” | Wheel nested under chassis rigid body            | Make `wheel_*` **siblings** of `base_link` under `perseverance` |
| Lab cannot find joints        | Names differ from USD                               | **Names are contracts** between Blender → USD → cfg |


---

## 3b.11 Suggested learning path on the NVIDIA machine


| Session | Do                                                         | Success looks like                                                 |
| ------- | ---------------------------------------------------------- | ------------------------------------------------------------------ |
| 1       | Open visual USD, add rigid bodies + box/cylinder colliders | Physics debug shapes visible; Play does not explode if held in air |
| 2       | Add one wheel joint + drive                                | That wheel spins in place when target velocity set                 |
| 3       | Add all six joints/drives                                  | Rover rolls forward on a flat ground prim                          |
| 4       | Attach one NavCam                                          | Viewport shows a sensible forward/mast view                        |
| 5       | Save articulated USD; sketch joint names on paper          | Ready to write `ArticulationCfg` in Chapter 2/7 coding sessions    |


---

## 3b.12 Mini glossary (Isaac-focused)


| Term                | Plain meaning                                                    |
| ------------------- | ---------------------------------------------------------------- |
| **Stage**           | The whole USD scene tree currently open                          |
| **Prim**            | One node in that tree                                            |
| **Xform**           | Pose frame (often from Blender Empty)                            |
| **Rigid body**      | PhysX-movable solid with mass                                    |
| **Collider**        | Shape used for contact                                           |
| **Revolute joint**  | Hinge: one rotational DOF                                        |
| **Drive**           | Motor on a joint (we use velocity targets)                       |
| **Articulation**    | Multi-link robot solved as one system                            |
| **DOF**             | Degree of freedom — an independent motion axis                   |
| **ArticulationCfg** | Lab’s Python recipe to spawn/control that robot                  |
| **Sensor prim**     | Camera, IMU, contact sensor, ray caster, etc.                    |
| **Decimation**      | How many physics steps per one RL action (Lab env setting later) |


---

## 3b.13 How this ties back to Blender choices


| Blender choice                | Pays off in Isaac when…                       |
| ----------------------------- | --------------------------------------------- |
| Six separate `wheel_*` meshes | Each can be its own link + joint              |
| Origin at hub                 | Revolute pivot is correct                     |
| `base_link` hierarchy         | Clear parent for joints                       |
| Camera Empties                | Known poses for Camera prims                  |
| Meters                        | Masses/forces have sane SI units              |
| No Blender rigid bodies       | No conflicting half-physics; Isaac owns PhysX |


---

## 3b.14 What we are *not* doing yet (so you are not lost)

- Full rocker-bogie passive joints  
- End-to-end vision PPO  
- Perfect Perseverance mass properties from NASA docs  
- ROS 2 bring-up

Those are later upgrades. **Rolling on flat ground with six driven wheels** is the milestone that unlocks the RL env.

---

## 3b.15 Next documents

- Finish Blender checklist in Chapter 3, then do Sessions 1–4 here.  
- **Chapter 2** (install/tooling) when you want the repo `pyproject.toml` + Lab install notes.  
- **Chapter 4** formalizes observation/action spaces once the robot can move.

When you finish “one wheel spins under a velocity drive,” tell me what you saw — that is the best mid-point check for Isaac, same as the Outliner check was for Blender.