"""CNN feature extractor for PPO on the flattened observation (needs ``ml-train``).

The observation is ``[raster (g*g), rays, scalars]`` (see ``flatten_observation``).
The raster goes through strided convolutions, so wall patterns are recognised
wherever they sit in the window, and the rays and scalars join before the policy
and value heads. It reads exactly the same observation as the MLP policy.
"""

from typing import Any

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

CONV_CHANNELS = (16, 32, 32)
FEATURES_DIM = 128


class MazeCNN(BaseFeaturesExtractor):
    """Convolutions over the wall raster, concatenated with rays and scalars."""

    def __init__(
        self,
        observation_space: Any,
        grid_size: int,
        features_dim: int = FEATURES_DIM,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.grid_size = grid_size
        first, second, third = CONV_CHANNELS
        self.convs = nn.Sequential(
            nn.Conv2d(1, first, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(first, second, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(second, third, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        reduced = ((grid_size + 1) // 2 + 1) // 2
        extras = int(observation_space.shape[0]) - grid_size * grid_size
        self.linear = nn.Sequential(
            nn.Linear(third * reduced * reduced + extras, features_dim), nn.ReLU()
        )

    def forward(self, observations: Any) -> Any:
        cells = self.grid_size * self.grid_size
        raster = observations[:, :cells].reshape(-1, 1, self.grid_size, self.grid_size)
        joined = torch.cat([self.convs(raster), observations[:, cells:]], dim=1)
        return self.linear(joined)
