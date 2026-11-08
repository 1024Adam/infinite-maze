"""A saved Stable-Baselines3 model as a ``Policy`` (needs the ``ml-train`` group)."""

from typing import Any

import numpy as np

from .observation import Observation, flatten_observation
from .policies import Policy


class ModelPolicy(Policy):
    """Deterministic actions from an SB3 model, on the shared flattened observation."""

    name = "model"

    def __init__(self, model: Any, name: str = "model") -> None:
        self.model = model
        self.name = name

    @classmethod
    def load(cls, path: str) -> "ModelPolicy":
        """Load a PPO model saved with ``model.save`` (``.zip``)."""
        from stable_baselines3 import PPO

        return cls(PPO.load(path, device="cpu"))

    def act(self, obs: Observation) -> int:
        flat = np.asarray(flatten_observation(obs), dtype=np.float32)
        action, _ = self.model.predict(flat, deterministic=True)
        return int(action)
