"""Custom RSL-RL actor-critic that fuses vector obs + a depth image.

Educational reference implementation — verify against your installed
rsl_rl.modules.ActorCritic before relying on it for a real run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_nn_activation


class DepthEncoder(nn.Module):
    """Small CNN: (N, H, W, 1) depth image -> (N, feature_dim)."""

    def __init__(self, height: int = 64, width: int = 64, feature_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 4 * 4, feature_dim)

    def forward(self, depth_img: torch.Tensor) -> torch.Tensor:
        # (N, H, W, 1) -> (N, 1, H, W) for Conv2d.
        x = depth_img.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class VisionActorCritic(ActorCritic):
    """Fuses ``vision.depth_cam`` with ``policy`` vector obs before the
    standard actor/critic MLP heads.

    Kept intentionally close to RSL-RL's own ``ActorCritic`` so the runner's
    checkpointing / logging code paths keep working unmodified.
    """

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, image_hw=(64, 64), **kwargs):
        # Reserve room in the MLP input size for the CNN's fused features.
        feature_dim = 64
        super().__init__(
            num_actor_obs=num_actor_obs + feature_dim,
            num_critic_obs=num_critic_obs + feature_dim,
            num_actions=num_actions,
            activation=kwargs.pop("activation", "elu"),
            **kwargs,
        )
        self.depth_encoder = DepthEncoder(*image_hw, feature_dim=feature_dim)

    def _fuse(self, vector_obs: torch.Tensor, depth_img: torch.Tensor) -> torch.Tensor:
        img_features = self.depth_encoder(depth_img)
        return torch.cat([vector_obs, img_features], dim=-1)

    def act(self, vector_obs, depth_img, **kwargs):
        return super().act(self._fuse(vector_obs, depth_img), **kwargs)

    def evaluate(self, vector_obs, depth_img, **kwargs):
        return super().evaluate(self._fuse(vector_obs, depth_img), **kwargs)
