"""A saved Stable-Baselines3 model as a ``Policy`` (needs the ``ml-train`` group)."""

from typing import Any

import numpy as np

from . import config
from .observation import Observation, flatten_observation, observation_size
from .policies import Policy


class ModelPolicy(Policy):
    """Deterministic actions from an SB3 model, on the shared flattened observation.

    The raster size must be the one the model was trained with; a mismatch raises
    instead of silently feeding the network a vector of the wrong length.
    """

    name = "model"

    def __init__(
        self,
        model: Any,
        name: str = "model",
        grid_size: int = config.OBS_GRID_SIZE,
        cell_px: int = config.OBS_GRID_CELL_PX,
    ) -> None:
        expected = int(model.observation_space.shape[0])
        if expected != observation_size(grid_size):
            raise ValueError(
                f"model expects {expected} observation values but grid_size "
                f"{grid_size} gives {observation_size(grid_size)}: pass the "
                f"observation settings the model was trained with"
            )
        self.model = model
        self.name = name
        self.grid_size = grid_size
        self.cell_px = cell_px

    @classmethod
    def load(
        cls,
        path: str,
        grid_size: int = config.OBS_GRID_SIZE,
        cell_px: int = config.OBS_GRID_CELL_PX,
    ) -> "ModelPolicy":
        """Load a PPO model saved with ``model.save`` (``.zip``)."""
        from stable_baselines3 import PPO

        return cls(PPO.load(path, device="cpu"), "model", grid_size, cell_px)

    def act(self, obs: Observation) -> int:
        flat = np.asarray(flatten_observation(obs), dtype=np.float32)
        action, _ = self.model.predict(flat, deterministic=True)
        return int(action)
