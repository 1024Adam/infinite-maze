"""Window-restricted planner: an information ceiling for a memoryless local view.

``WindowedPlanner`` is a diagnostic, not an agent. It plans on the same maze graph as
the oracle but may only use cells whose centres lie inside the ``grid_size`` x
``grid_size`` raster window around the player, and it keeps no state between
decisions (no committed target, no visited set). Each decision it walks the shortest
path to the visible reachable cell furthest to the right. That is the best a
memoryless policy with this window can do when the window shows the walls exactly,
so its score separates "the window lacks the information" from "the network cannot
learn it" (plan 4 R2). Lane keeping reuses the oracle's pixel-level controller.
"""

from collections import deque
from typing import Dict, Optional

from infinite_maze.core.simulation import Simulation

from . import config
from .oracle import _NEIGHBOUR_STEPS, Cell, MazeGeometry, OraclePolicy


class WindowedPlanner(OraclePolicy):
    """Memoryless planner restricted to the observation window (see module doc)."""

    name = "windowed"

    def __init__(
        self,
        grid_size: int = config.OBS_GRID_SIZE,
        cell_px: int = config.OBS_GRID_CELL_PX,
    ) -> None:
        super().__init__()
        self.gridSize = grid_size
        self.cellPx = cell_px
        self.sim: Optional[Simulation] = None

    def act_privileged(self, sim: Simulation) -> int:
        self.sim = sim
        return super().act_privileged(sim)

    def _visible(self, sim: Simulation, geometry: MazeGeometry, cell: Cell) -> bool:
        """Whether the centre of ``cell`` lies inside the raster window."""
        player = sim.player
        half = self.gridSize * self.cellPx / 2
        centreX = player.getX() + sim.getScrollOffset() + player.getWidth() / 2
        centreY = player.getY() + player.getHeight() / 2
        cellX = geometry.originX + cell[0] * geometry.cell + geometry.cell / 2
        cellY = geometry.originY + cell[1] * geometry.cell + geometry.cell / 2
        return abs(cellX - centreX) <= half and abs(cellY - centreY) <= half

    def next_cell(self, geometry: MazeGeometry, cell: Cell) -> Optional[Cell]:
        sim = self.sim
        assert sim is not None
        parent: Dict[Cell, Optional[Cell]] = {cell: None}
        depth = {cell: 0}
        queue = deque([cell])
        best = cell
        while queue:
            here = queue.popleft()
            if (here[0], -depth[here]) > (best[0], -depth[best]):
                best = here
            for dc, dr in _NEIGHBOUR_STEPS:
                other = (here[0] + dc, here[1] + dr)
                if other in parent or not geometry.passable(here, other):
                    continue
                if not self._visible(sim, geometry, other):
                    continue
                parent[other] = here
                depth[other] = depth[here] + 1
                queue.append(other)
        if best == cell:
            return None
        step = best
        while parent[step] != cell:
            previous = parent[step]
            assert previous is not None
            step = previous
        return step
