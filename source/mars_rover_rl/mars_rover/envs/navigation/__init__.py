"""Navigation tasks package — registers Gym ids on import."""

import gymnasium as gym

gym.register(
    id="Mars-Perseverance-Nav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "mars_rover.envs.navigation.config.navigation_env_cfg:NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": "mars_rover.envs.navigation.agents.rsl_rl_ppo_cfg:PerseverancePPORunnerCfg",
    },
)

gym.register(
    id="Mars-Perseverance-Nav-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "mars_rover.envs.navigation.config.navigation_env_cfg:NavigationRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "mars_rover.envs.navigation.agents.rsl_rl_ppo_cfg:PerseverancePPORunnerCfg",
    },
)
