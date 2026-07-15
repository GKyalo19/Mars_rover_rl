# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Perseverance Mars navigation RL environments for Isaac Lab.

This is the *root* of the importable package:

    import mars_rover
    print(mars_rover.__version__)

Keep this file free of `isaaclab` imports for now so you can still
`import mars_rover` on the Mac for packaging smoke tests.
Gym env registration will live under `mars_rover.envs` in a later chapter.
"""

__version__ = "0.1.0"
