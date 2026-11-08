"""Oracle: a privileged solver that reads the walls, used as the ceiling.

``OraclePolicy`` is not an observation-based policy. It reads the simulation's
lines and ``Simulation.getScrollOffset()``, builds the periodic cell graph of the
maze, finds the shortest path to a column a few cells ahead, and converts "go to
the next cell" into one of the five discrete actions at pixel level (it keeps the
10 px player centred in the 22 px corridor, and the real collision rules block
it, so it can never enter a wall). Game rules are not modified to help it.

World coordinates: ``world_x = player.x + scroll_offset``. The maze repeats every
``2 * MAZE_ROWS`` columns and the wrap between the last and first column is open in
every row. Columns left of the maze (the spawn area) are open space.
"""

from collections import deque
from typing import Dict, Optional, Set, Tuple

from infinite_maze.core.simulation import Simulation
from infinite_maze.utils.config import config as game_config

from . import config
from .policies import PrivilegedPolicy

Cell = Tuple[int, int]  # (column, row); columns are unrolled and may be negative

DO_NOTHING = game_config.get_movement_constant("DO_NOTHING")
RIGHT = game_config.get_movement_constant("RIGHT")
LEFT = game_config.get_movement_constant("LEFT")
UP = game_config.get_movement_constant("UP")
DOWN = game_config.get_movement_constant("DOWN")

# Neighbour preference when two moves are equally short.
_NEIGHBOUR_STEPS = ((1, 0), (0, -1), (0, 1), (-1, 0))


class MazeGeometry:
    """Grid dimensions and the wall sets of one maze tile."""

    def __init__(self) -> None:
        self.cell = game_config.MAZE_CELL_SIZE
        self.cols = 2 * max(game_config.MAZE_ROWS, 3)
        self.rows = max(game_config.MAZE_COLS, 3) - 1
        self.originX = game_config.PLAYER_START_X + self.cell
        self.originY = game_config.MAZE_TOP_Y
        self.period = self.cols * self.cell
        # (col, row): a wall on the left edge of that cell / the top edge of it
        self.verticalWalls: Set[Cell] = set()
        self.horizontalWalls: Set[Cell] = set()

    def load(self, sim: Simulation) -> None:
        """Read the tile's walls from the simulation's lines at their world x."""
        self.verticalWalls = set()
        self.horizontalWalls = set()
        offset = sim.getScrollOffset()
        for line in sim.getLines():
            tx = (line.getXStart() + offset - self.originX) % self.period
            ty = line.getYStart() - self.originY
            if tx % self.cell or ty % self.cell:
                raise ValueError(
                    "wall is not aligned to the cell grid; "
                    "check Simulation.getScrollOffset()"
                )
            key = (tx // self.cell, ty // self.cell)
            if line.getIsHorizontal():
                self.horizontalWalls.add(key)
            else:
                self.verticalWalls.add(key)

    def passable(self, a: Cell, b: Cell) -> bool:
        """Whether adjacent cells ``a`` and ``b`` are connected (no wall between)."""
        (ca, ra), (cb, rb) = a, b
        if not (0 <= ra < self.rows and 0 <= rb < self.rows):
            return False
        if ra == rb:  # horizontal neighbours
            right = max(ca, cb)  # the cell whose left edge is shared
            if right <= 0:
                return True  # spawn area, and the open entrance to the maze
            return (right % self.cols, ra) not in self.verticalWalls
        column = ca
        if column < 0:
            return True
        lower = max(ra, rb)
        return (column % self.cols, lower) not in self.horizontalWalls


class OraclePolicy(PrivilegedPolicy):
    """Shortest-path follower with pixel-level lane keeping (see module doc)."""

    name = "oracle"

    def __init__(
        self,
        lookahead_cols: int = config.ORACLE_LOOKAHEAD_COLS,
        align_tol_px: int = config.ORACLE_ALIGN_TOL_PX,
    ) -> None:
        self.lookahead = lookahead_cols
        self.alignTol = align_tol_px
        self.geometry: Optional[MazeGeometry] = None
        self.distances: Dict[int, Dict[Cell, int]] = {}
        self.target: Optional[int] = None

    def reset(self, seed: int) -> None:
        self.geometry = None
        self.distances = {}
        self.target = None

    def _geometry(self, sim: Simulation) -> MazeGeometry:
        if self.geometry is None:
            self.geometry = MazeGeometry()
            self.geometry.load(sim)
        return self.geometry

    def _distance_field(self, geometry: MazeGeometry, target: int) -> Dict[Cell, int]:
        """BFS distances (in cells) to column ``target``, within a window of columns."""
        cached = self.distances.get(target)
        if cached is not None:
            return cached
        low = target - self.lookahead - geometry.cols
        dist: Dict[Cell, int] = {(target, r): 0 for r in range(geometry.rows)}
        queue = deque(dist)
        while queue:
            cell = queue.popleft()
            for dc, dr in _NEIGHBOUR_STEPS:
                other = (cell[0] + dc, cell[1] + dr)
                if other in dist or not (low <= other[0] <= target):
                    continue
                if geometry.passable(cell, other):
                    dist[other] = dist[cell] + 1
                    queue.append(other)
        if len(self.distances) > config.ORACLE_CACHE_FIELDS:
            self.distances.clear()
        self.distances[target] = dist
        return dist

    def next_cell(self, geometry: MazeGeometry, cell: Cell) -> Optional[Cell]:
        """The neighbour on a shortest path to the committed target column, if any.

        The target stays fixed until the player is within half the lookahead of it:
        re-targeting every column makes two routes of equal length flip-flop.
        """
        if self.target is None or cell[0] > self.target - self.lookahead // 2:
            self.target = cell[0] + self.lookahead
        field = self._distance_field(geometry, self.target)
        here = field.get(cell)
        if here is None:  # fell outside the window (long backtrack): re-target
            self.target = cell[0] + self.lookahead
            field = self._distance_field(geometry, self.target)
            here = field.get(cell)
            if here is None:
                return None
        for dc, dr in _NEIGHBOUR_STEPS:
            other = (cell[0] + dc, cell[1] + dr)
            if field.get(other, here) < here and geometry.passable(cell, other):
                return other
        return None

    def current_cell(self, sim: Simulation, geometry: MazeGeometry) -> Cell:
        """The maze cell containing the centre of the player's box."""
        player = sim.player
        worldX = player.getX() + sim.getScrollOffset()
        column = int(
            (worldX + player.getWidth() / 2 - geometry.originX) // geometry.cell
        )
        row = int(
            (player.getY() + player.getHeight() / 2 - geometry.originY) // geometry.cell
        )
        return column, min(max(row, 0), geometry.rows - 1)

    def cells_to_column(self, sim: Simulation, column: int) -> Optional[int]:
        """Shortest path length (cells) from the player's cell to ``column``."""
        geometry = self._geometry(sim)
        field = self._distance_field(geometry, column)
        return field.get(self.current_cell(sim, geometry))

    def act_privileged(self, sim: Simulation) -> int:
        geometry = self._geometry(sim)
        cell = self.current_cell(sim, geometry)
        target = self.next_cell(geometry, cell)
        if target is None:
            return RIGHT

        player = sim.player
        size = geometry.cell
        # Top-left player position that centres the box in ``cell``.
        goalX = (
            geometry.originX + cell[0] * size + (size - player.getWidth()) // 2
        ) - sim.getScrollOffset()
        goalY = geometry.originY + cell[1] * size + (size - player.getHeight()) // 2
        errX = goalX - player.getX()
        errY = goalY - player.getY()

        if target[0] != cell[0]:  # horizontal move: lane-keep vertically first
            if abs(errY) > self.alignTol:
                return DOWN if errY > 0 else UP
            return RIGHT if target[0] > cell[0] else LEFT
        if abs(errX) > self.alignTol:
            return RIGHT if errX > 0 else LEFT
        return DOWN if target[1] > cell[1] else UP
