# Chapter 3 — Rover Asset Pipeline (Blender → USD)

> **Your role:** You make the Blender edits.  
> **Isaac’s role:** Physics (joints, drives, collisions, materials).  
> **Goal of this chapter:** A clean, named, meter-scale hierarchy that exports to USD without fighting you later.

---

## 3.0 Correcting the mental model (you are right)


| Layer                 | Where it happens                  | What it is                                                              |
| --------------------- | --------------------------------- | ----------------------------------------------------------------------- |
| **Visual meshes**     | Blender → USD                     | How the rover *looks*                                                   |
| **Hierarchy & names** | Blender (then refine in Isaac)    | Which mesh is “left front wheel”                                        |
| **Camera markers**    | Blender Empties (or Isaac Xforms) | Where HazCams / NavCams sit                                             |
| **Physics**           | **Isaac Sim / Isaac Lab**         | Rigid bodies, revolute joints, wheel drives, collision shapes, friction |
| **RL wiring**         | Isaac Lab `ArticulationCfg`       | Which joints PPO’s `[v, ω]` mapper commands                             |


So: **do not** try to build rocker-bogie PhysX in Blender.  
**Do** make the mesh file boring, tidy, and correctly named so Isaac can attach physics cleanly.

Think of Blender as preparing a mannequin. Isaac puts on the muscles (physics).

---

## 3.0.1 Concepts from this chapter (read before clicking)

These confused many people the first time — including the “Empty” and “Keep Transform” steps. Here is what they *are* and why *this project* needs them.

### Parenting (hierarchy)

**What:** Making object B a *child* of object A. When A moves/rotates, B follows.

**Why for us:** The rover must be one tree under a root (`perseverance` → `base_link` + `wheel_*` as **siblings**). Isaac joints connect **separate** rigid bodies; wheels must not be nested rigid bodies under the chassis. A pile of 39 unparented meshes is a visual diorama, not a robot.

**Analogy:** Your hand is parented to your forearm. Move the arm, the hand comes along.

### Empty

**What:** An object with **position + rotation + scale**, but **no mesh**. Invisible in the final render unless you draw its axes helper. In the Outliner it still has a name.

**Why for us:** HazCam/NavCam *meshes* are decorative plastic housings. The *sensor* in Isaac needs a precise pose (“sit here, look that way”). An Empty is a named thumbtack for that pose. Later Isaac places a real Camera prim on that thumbtack.

**Why not a Blender Camera?** You can use one, but Empties are simpler markers and avoid Blender camera lens settings that Isaac will ignore anyway. Isaac defines FOV/resolution in its own camera API.

**Project chain:**

```text
Blender Empty `cam_nav_left`
    → exports into USD as an Xform (a transform frame)
    → Isaac Camera prim uses that pose
    → observation pipeline (Phase B) reads images from that camera
```

### Keep Transform (when parenting)

**What:** Parenting option: “become a child, but **stay where you are in the world** right now.”

**Without Keep Transform:** Blender may reinterpret the child’s coordinates in the parent’s local space and the mesh can **jump** to a wrong place.

**With Keep Transform (`Ctrl+P` → Object Keep Transform):** The wheel stays on the axle visually; only the *family relationship* changes.

**Why for us:** You already placed wheels correctly in space. You only want to tell Blender “these belong under `base_link`,” not “recalculate my art.”

### Origin (orange dot)

**What:** The object’s local (0,0,0) — where Blender thinks the object’s “center” is.

**Why for us:** A wheel joint in Isaac spins around a point. If the origin is on the tire tread instead of the hub, the wheel orbits like a wrecking ball. **Origin to Geometry** (or to the axle) fixes that before export.

### More Isaac-side vocabulary

After Blender, continue with **[Chapter 3b — Isaac articulation & physics](03b_isaac_articulation_and_physics.md)** (Rigid Body, Revolute joint, Drive, colliders, `ArticulationCfg`), written the same “what / why / how it fits the rover” way.

---

## 3.1 What “ready for Isaac” means after Blender

When you finish this chapter’s Blender work, you should have:

1. Scene units = **Meters**
2. One root object (e.g. `perseverance` or `base_link`)
3. Six **separate** wheel meshes with clear names
4. Body / suspension visuals parented under the root (suspension can stay **one visual mesh** for v1)
5. **Empty** objects at camera locations (HazCams + NavCams), named clearly
6. No need for Blender cameras, armatures, or rigid bodies
7. Optional: junk hidden or deleted (name chips, ultra-fine cables) for a training LOD later

You will **not** yet have working driving physics. That comes after USD import in Isaac.

---



## 3.2 Blender survival kit (10 minutes)

Open Blender → **File → Open** →  
`/Users/gracekyalo/Downloads/Mars 2020 Perseverance Rover.blend`

### Viewport basics


| Action              | Shortcut / where                   |
| ------------------- | ---------------------------------- |
| Rotate view         | Middle-mouse drag                  |
| Pan                 | Shift + middle-mouse               |
| Zoom                | Scroll wheel                       |
| Select object       | Left-click                         |
| Box select          | B, then drag                       |
| Select all / none   | A / Alt+A                          |
| Delete              | X or Delete                        |
| Undo                | Cmd+Z                              |
| Search any operator | F3 (or Space, depending on keymap) |
| Switch to Edit Mode | Tab (with a mesh selected)         |
| Object Mode         | Tab again                          |




### Outliner (top-right)

This is your object list. You will live here for parenting and renaming.

- Click a name to select.
- Double-click a name to rename.
- Drag object onto another to **parent** (child moves with parent).



### Properties (bottom-right)

With an object selected, the tabs include:

- **Object Properties** (orange square) — name, location, parent  
- **Item** panel (N key in viewport) — transform numbers



### Save a working copy first

**File → Save As…** → save into the project, e.g.:

`~/Projects/Mars_rover_rl/source/mars_rover_rl/assets/robots/perseverance/perseverance_wip.blend`

Never edit only the Downloads original.

---



## 3.3 Step A — Units and scale check

1. **Scene Properties** (icon looks like a cone/scene) → **Units**
2. Set:
  - Unit System: **Metric**
  - Unit Scale: **1.0**
  - Length: **Meters**
3. Select `Wheels_objs` (Outliner).
4. Press **N** → **Item** → look at **Dimensions**.
  Wheel assembly height should be roughly **~0.5 m** (real Perseverance wheels are ~52.5 cm diameter). Your inspect already showed ~0.53 m — good.

If everything looked tiny or huge later in Isaac, this is the first place to re-check.

**Do not** Apply Scale randomly yet. We will apply transforms in a controlled step after parenting.

---



## 3.4 Step B — Ignore / remove what Isaac does not need



### Armature (animation skeleton)

Your file has an `Armature` with generic bones. It is **not** a physics tree.

Options (pick one):

- **Safe:** Select `Armature` → hide it (eye icon in Outliner), leave it alone.  
- **Cleaner:** Select `Armature` → X → Delete. If Blender warns about users, delete carefully or keep hidden.

Also fine to leave the scene `Lamp` / studio light; Isaac will use its own lighting.

### Blender rigid body

You have none — good. Do **not** add any.

---



## 3.5 Step C — Split the six wheels (important)

Right now `Wheels_objs` is **one mesh** containing six wheels (plus small bits). Isaac needs **six links**.

### C1 — Duplicate a backup

1. Select `Wheels_objs`
2. Shift+D, then right-click to cancel move (duplicate stays in place)
3. Rename the copy to `Wheels_objs_BACKUP`
4. Hide the backup (eye icon)



### C2 — Separate by loose parts

1. Select the working `Wheels_objs`
2. Tab → **Edit Mode**
3. Select all: **A**
4. **Mesh → Separate → By Loose Parts**
  (or F3 → search “Separate by Loose Parts”)
5. Tab → **Object Mode**

You should now see many new objects (inspect found **54** loose parts: 6 big wheels + hubs/cleats/etc.).

### C3 — Identify the six main wheels

1. Click each large piece in the viewport or Outliner.
2. The six main wheels are the big ones (inspect: six parts with ~3456 verts each).
3. Rename them exactly (recommended convention):


| Name       | Meaning      |
| ---------- | ------------ |
| `wheel_FL` | Front Left   |
| `wheel_FR` | Front Right  |
| `wheel_ML` | Middle Left  |
| `wheel_MR` | Middle Right |
| `wheel_RL` | Rear Left    |
| `wheel_RR` | Rear Right   |


**How to tell left/right/front:**

- In Blender, assume for now: look at the rover so the mast/cameras face **−Y** or **+Y** — pick a convention and stick to it.
- Perseverance’s “front” is the side with the big arm stowed and front HazCams on the body. Front HazCams mesh (`hazcams_front`) marks the **front**.
- Left/right = rover’s left/right (as if you sat in the rover facing forward), not your screen’s left.

Tip: select `hazcams_front`, note which end of the body it sits on — that end is **front**. Wheels nearest that end = `wheel_F`*.

### C4 — What about the tiny leftover pieces?

Hubcaps, bolts, tread extras that separated out:

- **Option A (simple):** Join small bits to the nearest wheel: select small bits, last-select the wheel, **Ctrl+J** (Join).  
- **Option B:** Delete tiny debris if it does not matter visually.  
- **Option C:** Parent small bits to the wheel without joining (keep as children).

For RL training visuals, A or B is enough.

---



## 3.6 Step D — Build a single root hierarchy

Isaac (and URDF/USD articulations) want a tree, not 39 floating roots.

### D1 — Create the root

1. Shift+A → **Empty → Plain Axes**
2. Rename to `base_link`
3. Move it to the approximate center of the chassis on the ground plane:
  - With Empty selected, **N** panel → set Location.  
  - A good start: X=0, Y=0, Z=0, then nudge so it sits under the body center.
4. Optional: Object Properties → Viewport Display → Size 0.5 so you can see it.



### D2 — Parent major parts under `base_link`

In Outliner, drag these onto `base_link` (or select child(ren), then select `base_link`, **Ctrl+P** → **Object (Keep Transform)**):

- `Body`, `Body_Parts`, and other body meshes you care about  
- `suspension` (visual only for v1)  
- all six `wheel_*`  
- `hazcams_front`, `hazcams_rear`, covers  
- mast chain pieces if you want them (`bottom`, `top`, `head`, `NavCams`, …)  
- OR parent whole groups gradually

**Keep Transform** is important so nothing jumps.

### D3 — Suggested target tree (v1) — important for Isaac physics

Blender parenting and PhysX rigid bodies are related but **not identical**.

For **visual organization** in Blender, this is fine:

```text
perseverance          ← root Empty/Xform (NO physics on this later)
├── base_link         ← chassis visuals (body + suspension meshes as children)
├── wheel_FL
├── wheel_FR
├── wheel_ML
├── wheel_MR
├── wheel_RL
├── wheel_RR
└── cam_… Empties
```

**Critical:** Make `wheel_*` **siblings** of `base_link` under `perseverance`, **not** children of `base_link`.

Why: In Isaac, each link that moves separately needs its **own** Rigid Body. A rigid body **cannot be the child of another rigid body**. If wheels sit under `base_link` and both have Rigid Body APIs, PhysX treats that as nested / the same body and you get errors like **“cannot create a joint between a body and itself.”**

So in Blender (best) or later in Isaac Stage (also fine):

1. Create root Empty named `perseverance`
2. Parent `base_link`, all `wheel_*`, and camera Empties to **`perseverance`** (Keep Transform)
3. Keep body/suspension meshes **under** `base_link` only (those stay visual parts of the chassis — they should **not** get their own Rigid Body in Isaac)

**Do not** joint wheels to `suspension` for v1. Suspension is decoration (or welded to the chassis). Joints are **chassis ↔ each wheel**.


---



## 3.7 Step E — Camera markers (Empties), not Blender cameras

Isaac will spawn real sensors. In Blender you only mark **poses**.

### E1 — Create an Empty at a camera

1. Select the mesh that shows where the camera is (e.g. `hazcams_front` or `NavCams`).
2. Shift+S → **Cursor to Selected**
3. Shift+A → **Empty → Plain Axes**
4. Rename, e.g. `cam_haz_front_left_outer`
5. Parent the Empty to `base_link` (Keep Transform)
6. Nudge the Empty so its origin sits roughly at the **lens** (use wireframe: Z or viewport overlays)

Repeat for the set you care about in Phase B:


| Empty name (suggestion)               | Role               |
| ------------------------------------- | ------------------ |
| `cam_haz_front_0` … `cam_haz_front_3` | Four front HazCams |
| `cam_haz_rear_0`, `cam_haz_rear_1`    | Two rear HazCams   |
| `cam_nav_left`, `cam_nav_right`       | Mast NavCams       |


**Orientation tip (do this carefully later in Isaac too):**  
Isaac cameras typically look along **−Z** with **+Y** up (USD/camera conventions vary by exporter). For now, aim the Empty’s local −Z toward the scene in front of that camera. You will verify look-direction after import with a test render in Isaac.

You can skip perfect stereo baselines until Chapter 8; markers are enough for the asset pipeline.

---



## 3.8 Step F — Optional cleanup (recommended)

Do these if the file feels messy:

1. **Join body visuals:** select multiple body meshes → Ctrl+J → rename `body_visual`.
2. **Hide science payload** you will not use in nav training (arm, PIXL, etc.): eye icon off, or move to a collection “DISABLED”.
3. **Origin to geometry** for each wheel (helps later joints):
  - Select `wheel_FL`
  - Right-click → **Set Origin → Origin to Geometry**  
  - Or Object → Set Origin → Origin to Geometry  
   Ideal origin ≈ wheel center (axle). If origin is wrong, joints in Isaac will orbit weirdly — fix origins now.

Check each wheel: in Object Mode, the orange origin dot should sit near the wheel hub center.

---



## 3.9 Step G — Apply transforms (once hierarchy looks right)

With **all rover objects selected** (or select `base_link` and children):

1. Confirm locations look correct.
2. **Ctrl+A** → **All Transforms** (or at least **Rotation & Scale**)

Do this **after** parenting and origin fixes. If something jumps, Undo and apply per-object more carefully (often **Scale** only first).

---



## 3.10 Step H — Export USD for Isaac



### H1 — Blender USD export

1. Select `base_link` and all descendants (select root → Outliner menu → select hierarchy, or use Select → Select Pattern).
  Easiest: in Outliner, click `base_link`, then **Select → Select Hierarchically** if available, or manually select children.
2. **File → Export → Universal Scene Description (.usd)**
  (Sometimes listed as `.usdc` / USDZ depending on Blender version.)
3. Export settings to prefer:
  - **Selected Objects** only  
  - **Meshes** on  
  - **Materials** on (optional)  
  - **Armatures** off / unused  
  - Apply transform-ish options if present: use world space consistently
4. Save as:

`~/Projects/Mars_rover_rl/source/mars_rover_rl/assets/robots/perseverance/perseverance_visual.usd`

### H2 — What you do next in Isaac (preview — not this Blender session)

On the NVIDIA machine, in Isaac Sim / Lab you will:

1. Open / reference the USD
2. Add **Rigid Body** APIs to `base_link` and each `wheel_*`
3. Add **Revolute joints** (axle axis) from chassis to each wheel
4. Add **Drive** on those joints (velocity targets)
5. Add **collision approximations** (cylinders for wheels, boxes for body)
6. Create **Camera** prims at your Empty poses
7. Wrap it in Isaac Lab `ArticulationCfg`

That is the physics step you correctly pointed at — **Isaac, not Blender**.

**Full beginner walkthrough (what each thing is + why + GUI steps):**  
→ **[Chapter 3b — Isaac articulation & physics](03b_isaac_articulation_and_physics.md)**

---



## 3.11 Acceptance checklist (before you call Blender “done”)

Tick these yourself:

- [ ] Working copy saved under `assets/robots/perseverance/`
- [ ] Units = Metric, meters, scale 1.0
- [ ] Six named wheels: `wheel_FL` … `wheel_RR`
- [ ] Each wheel origin near hub center
- [ ] Single root `base_link` with children parented (Keep Transform)
- [ ] `suspension` is visual child (not expected to articulate yet)
- [ ] Camera Empties created and named for HazCams / NavCams
- [ ] Armature ignored or deleted
- [ ] USD exported with selection hierarchy
- [ ] You did **not** rely on Blender rigid bodies for driving

---



## 3.12 Common beginner pitfalls


| Symptom                       | Likely cause                                    | Fix                                            |
| ----------------------------- | ----------------------------------------------- | ---------------------------------------------- |
| Object jumps when parenting   | Used wrong parent option                        | Undo; Ctrl+P → **Keep Transform**              |
| Wheels fly apart on separate  | Normal — they become many objects               | Rename/join leftovers                          |
| Everything tiny/huge in Isaac | Units or unapplied scale                        | Meters + Ctrl+A Scale                          |
| Wheel spins around rim        | Origin not at hub                               | Set Origin → Geometry / 3D Cursor at axle      |
| Cannot find objects           | Hidden collections                              | Outliner → disable filters; enable hidden eyes |
| Export missing pieces         | Exported selection only but children unselected | Select full hierarchy under `base_link`        |


---



## 3.13 Suggested session plan (don’t rush)


| Session | Do                                                                       |
| ------- | ------------------------------------------------------------------------ |
| 1       | Save WIP copy, set units, hide armature, explore Outliner                |
| 2       | Separate wheels, rename six wheels, fix origins                          |
| 3       | Create `base_link`, parent body/suspension/wheels                        |
| 4       | Add camera Empties, optional join/hide clutter                           |
| 5       | Apply transforms, export USD, screenshot Outliner tree for the repo docs |


When a session is done, tell me what you see (or paste Outliner names). I will help you sanity-check before you move to Isaac physics.

---



## 3.14 How this ties to RL (reminder)

- Blender gives **named wheel meshes** → Isaac joints → action mapper `[v, ω] → wheel speeds`.  
- Blender gives **camera Empties** → Isaac cameras → Phase B observations.  
- Blender does **not** define rewards, PPO, or rocker-bogie dynamics.

---

**Next after you finish Blender:** Chapter 2/3b — importing the USD into Isaac Lab and adding articulation (on the NVIDIA machine). For now, start with **Step A–B** (units + save WIP) and ping me when wheels are separated if you want a mid-point check.