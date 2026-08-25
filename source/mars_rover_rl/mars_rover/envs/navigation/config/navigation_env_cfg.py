"""Full navigation env configuration for Perseverance Baseline v1."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from mars_rover.assets.terrains.mars.mars_scene_cfg import MarsNavSceneCfg
from mars_rover.assets.terrains.mars.mars_terrain_cfg import MarsProceduralTerrainCfg
from mars_rover.envs.navigation.config.actions_cfg import ActionsCfg
from mars_rover.envs.navigation.config.commands_cfg import CommandsCfg
from mars_rover.envs.navigation.config.events_cfg import EventCfg
from mars_rover.envs.navigation.config.observations_cfg import ObservationsCfg
from mars_rover.envs.navigation.config.rewards_cfg import RewardsCfg
from mars_rover.envs.navigation.config.terminations_cfg import TerminationsCfg


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Perseverance navigation MDP on **flat** terrain (Baseline v1 default).

    Action dim 6, observation dim 19. Rough terrain is ``NavigationRoughEnvCfg``.
    """

    scene: MarsNavSceneCfg = MarsNavSceneCfg(num_envs=64, env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.sim.dt = 1.0 / 60.0
        self.episode_length_s = 60.0

        self.viewer.eye = (12.0, 12.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        if getattr(self.scene, "contact_sensor", None) is not None:
            self.scene.contact_sensor.update_period = self.sim.dt * self.decimation


@configclass
class NavigationRoughEnvCfg(NavigationEnvCfg):
    """Same MDP on seeded procedural rocks."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain = MarsProceduralTerrainCfg()
