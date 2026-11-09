"""Oracle-based potential shaping for the training reward (stdlib only).

The maze is periodic and fixed in world coordinates (see ``oracle.py``), so one
BFS distance field from spawn to a column the player cannot reach within
``max_steps`` stays valid for the whole episode. ``PotentialField`` holds it and
``PotentialShaper`` turns it into the per-decision shaping term

    F = weight * (gamma * phi(after) - phi(before)),   phi(cell) = -distance(cell)

The potential is privileged information: it is used only in the training reward
and is never part of the observation. Evaluation and baselines are unshaped.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

from infinite_maze.core.simulation import Simulation

from .oracle import Cell, MazeGeometry, distance_field

# Columns kept left of the spawn cell, and slack right of the farthest reachable one.
FIELD_LEFT_MARGIN_COLS: int = 6
FIELD_RIGHT_MARGIN_COLS: int = 4


@dataclass(frozen=True)
class ShapingConfig:
    """Potential shaping parameters; ``weight`` 0 disables shaping."""

    weight: float = 0.0  # reward per cell of path distance
    gamma: float = 0.995  # must equal ppo.gamma (checked by ExperimentConfig)


class PotentialField:
    """Static cell-distance field for one episode, read from a fresh Simulation."""

    def __init__(self, sim: Simulation, max_steps: int) -> None:
        geometry = MazeGeometry()
        geometry.load(sim)
        self.geometry = geometry
        startColumn = geometry.cell_of(sim)[0]
        # The player moves at most ``speed`` pixels of world x per tick.
        reach = math.ceil(max_steps * sim.player.getSpeed() / geometry.cell)
        self.low = startColumn - FIELD_LEFT_MARGIN_COLS
        self.target = startColumn + reach + FIELD_RIGHT_MARGIN_COLS
        self.distances: Dict[Cell, int] = distance_field(
            geometry, self.low, self.target
        )

    def potential(self, sim: Simulation) -> float:
        """``-d(cell)`` for the player's cell; raises if the cell is off the field."""
        cell = self.geometry.cell_of(sim)
        distance = self.distances.get(cell)
        if distance is None:
            raise ValueError(
                f"cell {cell} is outside the potential field "
                f"(columns {self.low}..{self.target}) or unreachable"
            )
        return -float(distance)


class PotentialShaper:
    """Per-episode shaping term, shared by ``MazeEnv`` and the ordering gate."""

    def __init__(self, shaping: ShapingConfig, max_steps: int) -> None:
        self.config = shaping
        self.maxSteps = max_steps
        self.field: Optional[PotentialField] = None
        self.previous = 0.0
        self.start = 0.0  # potential at reset (0 when shaping is off)

    @property
    def enabled(self) -> bool:
        return self.config.weight > 0.0

    def reset(self, sim: Simulation) -> None:
        """Start an episode: build the field from the freshly reset simulation."""
        if not self.enabled:
            return
        self.field = PotentialField(sim, self.maxSteps)
        self.start = self.previous = self.field.potential(sim)

    def step(self, sim: Simulation, terminated: bool) -> float:
        """Shaping reward for one decision (``repeat_action`` call).

        The terminating transition gets no shaping term, so ``death_penalty`` alone
        punishes death and dying can never be rewarded for ending the episode.
        """
        if not self.enabled or terminated:
            return 0.0
        if self.field is None:
            raise RuntimeError("call reset() before step()")
        after = self.field.potential(sim)
        term = self.config.weight * (self.config.gamma * after - self.previous)
        self.previous = after
        return term
