"""Gymnasium environment over ``Simulation`` (needs the ``ml-env`` group).

``MazeEnv`` reuses the shared pieces: ``build_observation`` +
``flatten_observation`` for observations, ``RewardFunction`` for rewards and
``repeat_action`` for action repeat, so training, evaluation and baselines
cannot drift apart.
"""

import os

# Must be set before pygame is imported by infinite_maze.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from typing import Any, Dict, Optional, Sequence, Tuple  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402

from infinite_maze.core.simulation import Simulation  # noqa: E402
from infinite_maze.utils.config import config as game_config  # noqa: E402

from . import config  # noqa: E402
from .observation import build_observation  # noqa: E402
from .observation import flatten_observation, observation_size  # noqa: E402
from .reward import RewardConfig, RewardFunction  # noqa: E402
from .stepping import repeat_action  # noqa: E402

NUM_ACTIONS = len(game_config.MOVEMENT_CONSTANTS)


class MazeEnv(gym.Env):
    """One maze episode per ``reset``; ``step`` repeats the action ``frame_skip`` ticks.

    The maze seed is drawn from ``seeds`` with Gymnasium's ``np_random`` (a single
    seed gives the fixed-seed experiment) and always passed explicitly to
    ``Simulation.reset``. ``options={"maze_seed": s}`` forces a specific seed.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        seeds: Sequence[int],
        start_pace: int = config.EVAL_START_PACE,
        max_steps: int = config.EVAL_MAX_STEPS,
        frame_skip: int = config.DEFAULT_FRAME_SKIP,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__()
        if not seeds:
            raise ValueError("seeds must not be empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive so every episode ends")
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        rewardConfig = reward_config if reward_config is not None else RewardConfig()
        rewardConfig.validate(max_steps)

        self.seeds = [int(s) for s in seeds]
        self.start_pace = start_pace
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.reward_config = rewardConfig
        self._reward = RewardFunction(rewardConfig)
        self._sim: Optional[Simulation] = None

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(observation_size(),), dtype=np.float32
        )

    def _observe(self, sim: Simulation) -> np.ndarray:
        flat = flatten_observation(build_observation(sim))
        observation: np.ndarray = np.asarray(flat, dtype=np.float32)
        return observation

    def _info(self, sim: Simulation) -> Dict[str, Any]:
        return {
            "score": sim.game.getScore(),
            "ticks": sim.getTicks(),
            "pace": sim.game.getPace(),
            "maze_seed": sim.getSeed(),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and "maze_seed" in options:
            mazeSeed = int(options["maze_seed"])
        else:
            mazeSeed = self.seeds[int(self.np_random.integers(len(self.seeds)))]

        if self._sim is None:
            self._sim = Simulation(
                seed=mazeSeed, start_pace=self.start_pace, max_steps=self.max_steps
            )
        else:
            self._sim.reset(mazeSeed)
        return self._observe(self._sim), self._info(self._sim)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._sim is None:
            raise RuntimeError("call reset() before step()")
        sim = self._sim
        outcome = repeat_action(sim, int(action), self._reward, self.frame_skip)
        return (
            self._observe(sim),
            outcome.reward,
            outcome.terminated,
            outcome.truncated,
            self._info(sim),
        )

    @property
    def simulation(self) -> Simulation:
        """The underlying simulation (after the first ``reset``)."""
        if self._sim is None:
            raise RuntimeError("call reset() first")
        return self._sim
