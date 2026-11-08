"""Baseline policies: a yardstick any learned policy has to beat."""

import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, NamedTuple, Optional

from infinite_maze.core.simulation import Simulation
from infinite_maze.utils.config import config as game_config

from . import config
from .observation import Observation

DO_NOTHING = game_config.get_movement_constant("DO_NOTHING")
RIGHT = game_config.get_movement_constant("RIGHT")
LEFT = game_config.get_movement_constant("LEFT")
UP = game_config.get_movement_constant("UP")
DOWN = game_config.get_movement_constant("DOWN")
ACTIONS = tuple(sorted(game_config.MOVEMENT_CONSTANTS.values()))


class Policy(ABC):
    """Maps observations to discrete actions (the ``MOVEMENT_CONSTANTS`` space)."""

    name = "policy"

    def reset(self, seed: int) -> None:
        """Prepare for a new episode on the maze generated from ``seed``."""

    @abstractmethod
    def act(self, obs: Observation) -> int:
        """Choose an action for ``obs``."""


class PrivilegedPolicy(Policy):
    """A policy that reads the simulation itself (walls included), not an
    observation. Used for the oracle ceiling; ``evaluate()`` calls
    ``act_privileged`` instead of ``act`` for these."""

    def act(self, obs: Observation) -> int:
        raise TypeError(f"{self.name} is privileged: use act_privileged(sim)")

    @abstractmethod
    def act_privileged(self, sim: Simulation) -> int:
        """Choose an action from the full simulation state."""


class RandomPolicy(Policy):
    """Uniformly random action; reseeded per episode so results are reproducible."""

    name = "random"

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng if rng is not None else random.Random(0)

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)

    def act(self, obs: Observation) -> int:
        return self.rng.choice(ACTIONS)


class AlwaysRightPolicy(Policy):
    """Always pushes right: the lower bound for "do nothing clever"."""

    name = "right"

    def act(self, obs: Observation) -> int:
        return RIGHT


class Probe(NamedTuple):
    """What the heuristic needs to know about the surroundings."""

    right_open: bool
    left_open: bool
    up_open: bool
    down_open: bool
    up_room: float  # how much room there is going up (comparable with down_room)
    down_room: float


class HeuristicPolicy(Policy):
    """Reactive baseline, not a maze solver: go right when possible.

    When blocked it moves up or down toward the side with more room, and commits
    to that direction for a number of steps (hysteresis) so it does not
    oscillate. It is expected to get stuck in dead ends.
    """

    name = "heuristic"

    def __init__(
        self,
        open_px: int = config.HEURISTIC_OPEN_PX,
        commit_steps: int = config.HEURISTIC_COMMIT_STEPS,
        ray_cap_px: int = config.OBS_RAY_CAP_PX,
    ) -> None:
        self.openFraction = open_px / ray_cap_px
        self.commitSteps = commit_steps
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.direction: Optional[int] = None
        self.remaining = 0

    def _probe(self, obs: Observation) -> Probe:
        right, left, up, down = obs.rays
        threshold = self.openFraction
        return Probe(
            right_open=right > threshold,
            left_open=left > threshold,
            up_open=up > threshold,
            down_open=down > threshold,
            up_room=up,
            down_room=down,
        )

    def act(self, obs: Observation) -> int:
        probe = self._probe(obs)
        if probe.right_open:
            self.direction = None
            self.remaining = 0
            return RIGHT

        if self.direction is not None and self.remaining > 0:
            stillOpen = probe.up_open if self.direction == UP else probe.down_open
            if stillOpen:
                self.remaining -= 1
                return self.direction

        if not probe.up_open and not probe.down_open:
            self.direction = None
            self.remaining = 0
            return LEFT if probe.left_open else DO_NOTHING

        if probe.up_open and probe.down_open:
            if probe.up_room > probe.down_room:
                choice = UP
            elif probe.down_room > probe.up_room:
                choice = DOWN
            else:
                choice = self.direction if self.direction is not None else UP
        else:
            choice = UP if probe.up_open else DOWN
        self.direction = choice
        self.remaining = self.commitSteps - 1
        return choice


class RasterHeuristicPolicy(HeuristicPolicy):
    """Same logic as ``HeuristicPolicy`` but reading only the wall raster.

    Used to check whether the raster adds anything over the rays.
    """

    name = "heuristic-raster"

    def _probe(self, obs: Observation) -> Probe:
        grid = obs.grid
        size = len(grid)
        center = size // 2

        def room(rowStep: int, colStep: int) -> int:
            count = 0
            row, col = center + rowStep, center + colStep
            while 0 <= row < size and 0 <= col < size and grid[row][col] == 0:
                count += 1
                row += rowStep
                col += colStep
            return count

        up, down = room(-1, 0), room(1, 0)
        return Probe(
            right_open=room(0, 1) > 0,
            left_open=room(0, -1) > 0,
            up_open=up > 0,
            down_open=down > 0,
            up_room=float(up),
            down_room=float(down),
        )


def make_policy(name: str) -> Policy:
    """Create a baseline policy by CLI name."""
    factories: Dict[str, Callable[[], Policy]] = {
        "random": RandomPolicy,
        "right": AlwaysRightPolicy,
        "heuristic": HeuristicPolicy,
        "heuristic-raster": RasterHeuristicPolicy,
    }
    if name not in factories:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(factories)}")
    return factories[name]()
