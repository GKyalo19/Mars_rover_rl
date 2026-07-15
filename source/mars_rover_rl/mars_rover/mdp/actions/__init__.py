# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for Perseverance.

Do not eager-import Isaac-dependent modules here — that would break
``import mars_rover`` on a Mac without Isaac Lab.

On the NVIDIA machine, configs import directly, e.g.::

    from mars_rover.mdp.actions.actions_cfg import DifferentialDriveActionCfg
"""
