# Chapter 2 — Install, Tooling & Starting the Codebase

> **Mode:** Full code is provided below; your job is to **read every comment**, run the install, and be able to explain each block in your own words.  
> **Goal:** Make `Mars_rover_rl` an installable Isaac Lab **external project**, then prove `import mars_rover` works.  
> **Stack:** Isaac Sim 6.0 · Isaac Lab 3.0 · RSL-RL (PPO) · Perseverance on generated Mars terrain  

These packaging files are already in the repo (adapted from Isaac Lab’s official extension template). Use this chapter as the **annotated tour**.

---

## 2.0.0 Bigger picture — what did we actually “make installable”?

This section connects packaging to **your rover project** and to **how industry ships software**.

### The problem packaging solves

Your training code will eventually look like:

```python
import mars_rover
from mars_rover.envs.navigation... import ...
```

Python only finds `mars_rover` if that package is on the **module search path**.  
Packaging says: “Here is a named library; install it into this Python environment so any script can import it.”

Without that, every script needs brittle hacks like `sys.path.append("/Users/.../source/...")`. That breaks when you change machines, clone to the NVIDIA laptop, or share with a teammate. **Industry does not do that** for real projects — they ship **packages**.

### Two different “worlds” we are registering in

| World | File | What it answers | Who cares |
|-------|------|-----------------|-----------|
| **Python / pip** | `setup.py` + `pyproject.toml` | “How do I `import mars_rover`?” | Your scripts, tests, RSL-RL train entrypoints |
| **Isaac Sim / Omniverse** | `config/extension.toml` | “What is this extension, and which Lab extensions must load first?” | Isaac Sim’s extension/plugin loader |

Same project, **two ID cards**:

- Pip ID → Python ecosystem (like `numpy`, `torch`, any company internal library).  
- Extension ID → NVIDIA’s app plugin system (same *idea* as a VS Code extension manifesto or a ROS `package.xml`: declare yourself to the platform).

### What each file *achieves* for Perseverance RL

| File | Defines | Achieves for *this* project |
|------|---------|------------------------------|
| **`extension.toml`** | Title, version, Isaac deps (`isaaclab`, …), Python module name | Tells Isaac: “Lab-related extension; module is `mars_rover`; load Lab first.” |
| **`setup.py`** | Pip name, which folder to install, light pip deps, Python version | Makes `pip install -e …` put `mars_rover` on the import path for `train.py`. |
| **`pyproject.toml`** | Build backend (setuptools) | Tells modern pip *how* to build/install (PEP 517/518 industry standard). |
| **`mars_rover/__init__.py`** | Package root + `__version__` | Proves the library exists; later holds public API. |

None of these train the rover. They are the **loading dock** so training code can be found and run the same way everywhere.

### Local vs remote — can someone else use this?

**What you did with `pip install -e` is local only.**

| Piece | Meaning |
|-------|---------|
| `pip install` | Register this package into *this* Python environment (`.venv` or Lab conda) |
| `-e` (**editable**) | Don’t copy source into `site-packages`; **link** back to your project folder so edits apply immediately |
| `source/mars_rover_rl` | A path on **your disk** |

So:

- ✅ Works on **your Mac** in that env  
- ❌ Does **not** appear on a friend’s laptop by itself  
- ❌ Is **not** on PyPI / the public internet  
- ❌ Is **not** on GitHub until you create a remote and `git push`

**How industry (and you) share code:**

```text
1) GitHub (normal for research)
   Other machine:  git clone … && pip install -e source/mars_rover_rl

2) Optional later: PyPI
   Anyone:  pip install some_name
   (We are NOT publishing yet.)
```

**Your workflow:**

```text
Mac: edit → commit → push GitHub
NVIDIA: git pull → pip install -e (into Lab’s env) → train
```

Each machine installs into its **own** environment. The thing shared remotely is **git**, not the editable install.

### Editable install — industry mental model

| Mode | When used |
|------|-----------|
| **Editable (`-e`)** | Daily research — change rewards, re-run train, no reinstall |
| **Normal install** | CI / frozen releases |
| **`pip install git+https://…`** | Install from GitHub without manual clone |
| **PyPI** | Public or company package index |

Robotics/ML teams almost always use **editable + git** during research (Isaac Lab external projects, ROS workspaces, internal PyTorch libs — same pattern).

### How this sits in the full Perseverance stack

```text
GitHub repo  ← shared source of truth
    → packaging (Ch.2)  ← import mars_rover
    → assets / envs / mdp (Ch.3–6)  ← rover, Mars, obs, rewards
    → scripts/train.py + RSL-RL (Ch.7)  ← actually learn navigation
```

Chapter 2 is **step zero of shipping a research codebase the way labs do**: named package, reproducible install, git for distribution, platform manifest for Isaac.

### Why both `extension.toml` and `setup.py`?

- **Pip alone** → enough for `import mars_rover` in plain Python.  
- **Isaac Sim** loads **extensions**; `extension.toml` declares “I am an extension; I need `isaaclab`; my module is `mars_rover`.”

| Domain | Manifest |
|--------|----------|
| Python library | `pyproject.toml` / `setup.py` |
| Isaac / Omniverse | `extension.toml` |
| ROS | `package.xml` |
| Node | `package.json` |

Universal idea: **manifest + installable package**. You are learning the Isaac flavor.

### One sentence to keep

**Packaging does not train Perseverance — it makes your navigation code a real, importable library that any machine (after cloning) can install the same way industry installs internal ML/robotics packages.**

---

## 2.0 Zoomed out — what we are actually building

```text
generated Mars terrain + obstacles + random goal
        │
        ▼
Perseverance senses (obs) ──▶ PPO policy ──▶ [v, ω] ──▶ wheel drives
        │                         ▲
        └──── reward / done ──────┘
```

**Phase A “path planning”** = PPO learning a safe reactive policy from rewards/obs (not a separate A module yet).  
**Inspiration:** [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab) patterns — not a copy.

---



## 2.1 Coding arc (chapters ↔ files)


| Chapter     | Milestone                                      |
| ----------- | ---------------------------------------------- |
| **2 (now)** | Package installs; `import mars_rover_rl` works |
| **3 / 3b**  | Rover USD articulated in Sim                   |
| **4**       | Obs + `[v, ω]` action wiring                   |
| **5**       | Rewards + terminations                         |
| **6**       | Mars terrain + goals (RLRoverLab-inspired)     |
| **7**       | RSL-RL PPO `train.py`                          |
| **8**       | Vision / reach-avoid (optional)                |
| **9**       | Evaluation / play                              |


---



## 2.2 Two machines


| Machine    | Job                                  |
| ---------- | ------------------------------------ |
| **Mac**    | Edit docs/code, packaging smoke test |
| **NVIDIA** | Isaac Lab install, training, Play    |


---



## 2.3 Mental model: Project → Extension → Task


| Layer              | Plain English             | Path                                  |
| ------------------ | ------------------------- | ------------------------------------- |
| **Project**        | Git repo                  | `Mars_rover_rl/`                      |
| **Extension**      | Installable Lab extension | `source/mars_rover_rl/`               |
| **Python package** | What you `import`         | `source/mars_rover_rl/mars_rover_rl/` |
| **Task**           | Gym env id (later)        | e.g. `Mars-Perseverance-Nav-v0`       |


`pip install -e source/mars_rover_rl` teaches Python where `mars_rover_rl` lives while you keep editing.

---



## 2.4 Folder layout (why two `mars_rover_rl` names)

```text
source/mars_rover_rl/                 ← extension root (pip install -e targets this)
├── config/extension.toml
├── pyproject.toml
├── setup.py
├── docs/README.md
└── mars_rover_rl/                    ← importable package
    ├── __init__.py
    ├── assets/
    ├── envs/
    ├── mdp/
    ├── sensors/
    └── utils/
```


| Path                                  | Role                               |
| ------------------------------------- | ---------------------------------- |
| `source/mars_rover_rl/`               | Extension project (has `setup.py`) |
| `source/mars_rover_rl/mars_rover_rl/` | Module you import                  |


`setup(packages=["mars_rover_rl"])` means “install the **subfolder** named `mars_rover_rl`.” That is why nesting exists — same pattern as upstream `isaaclab_tasks`.

---



## 2.5 `extension.toml` — full file + explanations

**File:** `source/mars_rover_rl/config/extension.toml`  
**Audience:** Isaac Sim extension manager (not pip).  
**Analogy:** Passport for Omniverse — who you are, version, which other extensions must board the plane first.

```toml
[package]

# Semantic Versioning: MAJOR.MINOR.PATCH (https://semver.org/)
# 0.1.0 = early development; bump MINOR when we add features, PATCH for fixes.
version = "0.1.0"

# Human-readable name shown in Isaac / Omniverse extension UIs.
title = "Mars Rover RL"

# Who maintains this extension (change to your name/email anytime).
author = "Grace Kyalo"
maintainer = "Grace Kyalo"

# One-line summary of what this package does.
description = "Perseverance Mars navigation RL environments for Isaac Lab 3 / Isaac Sim 6."

# Optional path to a readme relative to this extension folder.
readme = "docs/README.md"

# Your future GitHub URL (update when you create the remote).
repository = "https://github.com/gracekyalo/Mars_rover_rl"
category = "isaaclab"

# Search tags — not used by Python imports, just metadata.
keywords = ["isaaclab", "mars", "rover", "perseverance", "rl", "ppo", "navigation"]

[dependencies]
# These are *Isaac Sim extension* dependencies (not pip packages).
# Empty {} means "require this extension to be enabled; any compatible version".
# Isaac Lab must already be installed on the NVIDIA machine.
"isaaclab" = {}
"isaaclab_assets" = {}
"isaaclab_rl" = {}
"isaaclab_tasks" = {}

[core]
# If true, Omniverse can hot-reload the extension while the app runs.
# false is safer while we are still learning / changing APIs often.
reloadable = false

[[python.module]]
# THIS is the import name: `import mars_rover_rl`
# It must match the nested folder: source/mars_rover_rl/mars_rover_rl/
name = "mars_rover_rl"

[isaac_lab_settings]
# Uncomment later only if you need system packages or a ROS workspace.
# apt_deps = ["example_package"]
# ros_ws = "path/from/extension_root/to/ros_ws"
```


| Block                  | Meaning                                          |
| ---------------------- | ------------------------------------------------ |
| `[package]`            | Title, version, author — also read by `setup.py` |
| `[dependencies]`       | Other **extensions** that must load first        |
| `"isaaclab" = {}`      | Core Lab (envs, managers, articulations)         |
| `"isaaclab_assets"`    | Asset helpers                                    |
| `"isaaclab_rl"`        | RL glue toward RSL-RL                            |
| `"isaaclab_tasks"`     | Official task ecosystem utilities                |
| `[[python.module]]`    | Declares import name (must match folder)         |
| `[isaac_lab_settings]` | Optional apt/ROS — ignore for now                |


**Learning check:** Why `{}` after `"isaaclab"`?  
→ “Depend on this extension; don’t pin a special version object here.”

---



## 2.6 `setup.py` — full file + explanations

**File:** `source/mars_rover_rl/setup.py`  
**Audience:** **pip**.  
**Analogy:** Shipping label — package name, which folder to include, light pip deps.

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'mars_rover_rl' Python package.

Run from the repo root (on the NVIDIA machine, inside Isaac Lab's Python):

    python -m pip install -e source/mars_rover_rl

The `-e` means *editable*: you can change code without reinstalling.
"""

import os

import toml
from setuptools import setup

# Absolute path to this file's directory (= the extension root).
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

# Load metadata from extension.toml so we do not duplicate version/description.
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Pip dependencies installed WITH this package.
# Keep this light: Isaac Lab / torch / isaacsim are already provided by your
# Lab conda (or Isaac) environment on the NVIDIA machine. Do NOT pin isaaclab
# here the way older third-party repos sometimes do — it fights Lab 3 installs.
INSTALL_REQUIRES = [
    "psutil",  # small utility dep used in many Lab extensions; harmless default
    "toml",  # so setup can always read extension.toml (also a build need)
]

setup(
    # Pip / import distribution name.
    name="mars_rover_rl",
    # Folders under this extension root that contain Python packages.
    # Matches: source/mars_rover_rl/mars_rover_rl/
    packages=["mars_rover_rl"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    # Include non-.py files declared by package data / MANIFEST if we add any.
    include_package_data=True,
    # Isaac Lab 3 expects modern Python (3.10+).
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 6.0.0",
    ],
    # zip_safe=False: prefer extracted files on disk (extensions + assets).
    zip_safe=False,
)
```


| Piece                        | Meaning                                               |
| ---------------------------- | ----------------------------------------------------- |
| `toml.load(extension.toml)`  | One source of truth for version/description           |
| `INSTALL_REQUIRES`           | Extra pip pkgs *beyond* Lab                           |
| No `isaaclab==…` pin         | Lab comes from your Sim/Lab env; pinning fights Lab 3 |
| `packages=["mars_rover_rl"]` | Install nested package folder                         |
| `python_requires`            | Need 3.10+                                            |
| `zip_safe=False`             | Keep real filesystem paths for assets later           |
| `-e`                         | Editable install                                      |


**vs RLRoverLab:** they pin Lab 2.2 + skrl. We follow **Lab 3 template** + RSL-RL later.

---



## 2.7 `pyproject.toml` — full file + explanations

**File:** `source/mars_rover_rl/pyproject.toml`  
**Audience:** build tools (pip). Tiny on purpose — metadata stays in `setup.py` / `extension.toml` (Lab style).

```toml
# Tells pip/build tools HOW to build this package.
# We still use setup.py for the actual metadata (Isaac Lab template style).

[build-system]
# setuptools: the classic Python packaging library.
# wheel: builds installable wheel archives.
# toml: needed because setup.py reads config/extension.toml.
# setuptools pinned <82 to match Isaac Lab's own template (compatibility).
requires = ["setuptools<82.0.0", "wheel", "toml"]
build-backend = "setuptools.build_meta"
```


| Key             | Meaning                             |
| --------------- | ----------------------------------- |
| `requires`      | Tools needed to *build* the install |
| `build-backend` | “Use setuptools / our setup.py”     |


---



## 2.8 Package `__init__.py` — full file + explanations

**File:** `source/mars_rover_rl/mars_rover_rl/__init__.py`

```python
# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Perseverance Mars navigation RL environments for Isaac Lab.

This is the *root* of the importable package:

    import mars_rover_rl
    print(mars_rover_rl.__version__)

Keep this file free of `isaaclab` imports for now so you can still
`import mars_rover_rl` on the Mac for packaging smoke tests.
Gym env registration will live under `mars_rover_rl.envs` in a later chapter.
"""

__version__ = "0.1.0"
```

Keep `__version__` in sync with `extension.toml`’s `version`.

---



## 2.9 Install & smoke test (you run this)



### Mac

Homebrew Python blocks system-wide pip (PEP 668). Use a venv:

```bash
cd ~/Projects/Mars_rover_rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e source/mars_rover_rl
python -c "import mars_rover_rl; print(mars_rover_rl.__version__)"
```

(`.venv/` is gitignored — local only.)

### NVIDIA (Isaac Lab’s Python)

Same commands inside the Lab conda/env. If needed: `python -m pip install toml`.

**Pass:** prints `0.1.0`.

---



## 2.10 Scripts folder

See `scripts/README.md`. Implement `train.py` / `play.py` / `zero_agent.py` only in later chapters.

---



## 2.11 RLRoverLab study guide (still useful)


| Look at                 | Steal the *idea*                     | Don’t copy blindly         |
| ----------------------- | ------------------------------------ | -------------------------- |
| `mdp/rewards.py`        | Progress, collision, smooth, reverse | Exact weights / Lab 2 APIs |
| `mdp/observations.py`   | Distance, heading, height scan       | Camera-first as default    |
| `assets/terrains/mars/` | Mars look / generation ideas         | Their robots               |
| Agents / learning       | Need an RL config object             | skrl — we use **RSL-RL**   |


Write ≤5 bullets: “Ideas I want for Perseverance.”

---



## 2.12 Path planning & PPO in the roadmap

```text
Ch 4–5 → what “safe progress” means (obs + reward)
Ch 6   → Mars world
Ch 7   → PPO optimizes πθ(a|o)
Ch 8   → optional explicit reach-avoid
```

---



## 2.13 Checklist

- [ ] I can explain why there are two folders named `mars_rover_rl`
- [ ] I can explain `[dependencies]` vs `INSTALL_REQUIRES` (extension deps vs pip deps)
- [ ] I read every comment in `extension.toml` / `setup.py`
- [ ] Smoke test: `import mars_rover_rl` → `0.1.0`
- [ ] ≤5 RLRoverLab inspiration bullets written

---



## 2.14 Your next action

1. Open the three files in the editor and read them with this chapter beside you.
2. Run the smoke-test commands.
3. Tell me: did import work? Any error text?
4. When that passes, we start **Chapter 4** — observation & action spaces — same style: **full starter code + deep explanations**.

---



## 2.15 North star

**Install the package so Isaac can see it; then teach Perseverance to reach goals on Mars-like terrain with PPO — and understand every file you touch.**