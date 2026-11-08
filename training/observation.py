"""Observation of a ``Simulation``: an egocentric wall raster, rays and scalars.

``build_observation`` is a pure function of the simulation state. The raster is
centred on the player and marks every cell a wall segment crosses, so it does not
depend on how far the lines have scrolled.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from infinite_maze.core.simulation import Simulation
from infinite_maze.utils.config import config as game_config

from . import config

Rays = Tuple[float, float, float, float]  # right, left, up, down
RAY_NAMES = ("right", "left", "up", "down")
SCALAR_NAMES = ("x", "y", "pace", "pace_phase")


@dataclass
class Observation:
    """What the agent sees at one step."""

    grid: List[List[int]]  # grid[row][col]; 1 = a wall crosses that raster cell
    rays: Rays  # distance to the nearest wall, as a fraction of the ray cap
    scalars: Tuple[float, ...]  # x, y, pace, pace phase (all in [0, 1])


def window_origin(
    sim: Simulation,
    grid_size: int = config.OBS_GRID_SIZE,
    cell_px: int = config.OBS_GRID_CELL_PX,
) -> Tuple[int, int]:
    """Top-left pixel of the raster window, centred on the player."""
    px, py = sim.player.getPosition()
    centerX = int(px) + sim.player.getWidth() // 2
    centerY = int(py) + sim.player.getHeight() // 2
    half = (grid_size * cell_px) // 2
    return centerX - half, centerY - half


def _single_index(coord: float, origin: int, cell_px: int, size: int) -> Optional[int]:
    index = int((coord - origin) // cell_px)
    return index if 0 <= index < size else None


def _index_range(
    start: float, end: float, origin: int, cell_px: int, size: int
) -> range:
    """Raster indices a half-open span ``[start, end)`` crosses, clipped to the grid."""
    first = int((start - origin) // cell_px)
    last = -int((origin - end) // cell_px) - 1  # ceil((end - origin) / cell) - 1
    last = max(first, last)
    return range(max(first, 0), min(last, size - 1) + 1)


def build_observation(
    sim: Simulation,
    grid_size: int = config.OBS_GRID_SIZE,
    cell_px: int = config.OBS_GRID_CELL_PX,
    ray_cap_px: int = config.OBS_RAY_CAP_PX,
) -> Observation:
    """Build the observation for the current state of ``sim``."""
    player = sim.player
    px, py = player.getPosition()
    left, top = px, py
    right = px + player.getWidth()
    bottom = py + player.getHeight()

    originX, originY = window_origin(sim, grid_size, cell_px)
    windowWidth = grid_size * cell_px
    # Only lines that can matter for the window or a ray are examined.
    xLow = min(originX, left - ray_cap_px)
    xHigh = max(originX + windowWidth, right + ray_cap_px)

    grid = [[0] * grid_size for _ in range(grid_size)]
    cap = float(ray_cap_px)
    toRight = toLeft = toUp = toDown = cap

    for line in sim.getLines():
        xs = line.getXStart()
        xe = line.getXEnd()
        if xe < xLow or xs > xHigh:
            continue
        ys = line.getYStart()
        ye = line.getYEnd()

        if line.getIsHorizontal():
            fixedRow = _single_index(ys, originY, cell_px, grid_size)
            if fixedRow is not None:
                for col in _index_range(xs, xe, originX, cell_px, grid_size):
                    grid[fixedRow][col] = 1
            # Closed intervals, matching the engine's collision checks: a wall
            # touching the player's box counts as being in the way.
            if xs <= right and xe >= left:
                if ys >= bottom:
                    toDown = min(toDown, ys - bottom)
                elif ys <= top:
                    toUp = min(toUp, top - ys)
            if top <= ys <= bottom:
                if xe > right:
                    toRight = min(toRight, max(0, xs - right))
                if xs < left:
                    toLeft = min(toLeft, max(0, left - xe))
        else:
            fixedCol = _single_index(xs, originX, cell_px, grid_size)
            if fixedCol is not None:
                for row in _index_range(ys, ye, originY, cell_px, grid_size):
                    grid[row][fixedCol] = 1
            if ys <= bottom and ye >= top:
                if xs >= right:
                    toRight = min(toRight, xs - right)
                elif xs <= left:
                    toLeft = min(toLeft, left - xs)
            if left <= xs <= right:
                if ye > bottom:
                    toDown = min(toDown, max(0, ys - bottom))
                if ys < top:
                    toUp = min(toUp, max(0, top - ye))

    rays = (toRight / cap, toLeft / cap, toUp / cap, toDown / cap)
    return Observation(grid=grid, rays=rays, scalars=_scalars(sim))


def _scalars(sim: Simulation) -> Tuple[float, ...]:
    game = sim.game
    px, py = sim.player.getPosition()
    xFraction = (px - game.X_MIN) / (game.X_MAX - game.X_MIN)
    yFraction = (py - game.Y_MIN) / (game.Y_MAX - game.Y_MIN)
    paceFraction = min(game.getPace() / config.OBS_PACE_CAP, 1.0)
    # Phase within the pace interval: independent of max_steps, so training and
    # evaluation episodes of different lengths see the same signal.
    paceInterval = game_config.PACE_UPDATE_TICKS
    pacePhase = (sim.getTicks() % paceInterval) / paceInterval
    return (
        min(max(xFraction, 0.0), 1.0),
        min(max(yFraction, 0.0), 1.0),
        paceFraction,
        pacePhase,
    )


def observation_size(
    grid_size: int = config.OBS_GRID_SIZE,
) -> int:
    """Length of ``flatten_observation`` output for a raster of ``grid_size``."""
    return grid_size * grid_size + len(RAY_NAMES) + len(SCALAR_NAMES)


def flatten_observation(obs: Observation) -> List[float]:
    """Flatten to ``[grid row-major..., rays..., scalars...]`` (all in [0, 1]).

    The single flattening shared by the environment, ``ModelPolicy`` and any
    demonstration export.
    """
    flat: List[float] = [float(cell) for row in obs.grid for cell in row]
    flat.extend(obs.rays)
    flat.extend(obs.scalars)
    return flat
