"""Small helpers for building hand-made simulations in the training tests."""

from typing import Iterable, Optional
from unittest.mock import patch

from infinite_maze.core.simulation import Simulation
from infinite_maze.entities.maze import Line

IDLE = (0, 0)
RIGHT, LEFT, UP, DOWN = (1, 0), (-1, 0), (0, -1), (0, 1)


def make_sim(
    lines: Optional[Iterable[Line]] = None,
    x: float = 200,
    y: float = 200,
    **kwargs: object,
) -> Simulation:
    """Simulation with a hand-made maze (generation is skipped for speed)."""
    with patch.object(Line, "generateMaze", return_value=[]):
        sim = Simulation(**kwargs)  # type: ignore[arg-type]
    sim.lines = list(lines or [])
    sim.player.setX(x)
    sim.player.setY(y)
    return sim


def vertical(x: int, y_start: int, y_end: int) -> Line:
    """A vertical wall segment at ``x``."""
    return Line((x, y_start), (x, y_end))


def horizontal(y: int, x_start: int, x_end: int) -> Line:
    """A horizontal wall segment at ``y``."""
    return Line((x_start, y), (x_end, y))
