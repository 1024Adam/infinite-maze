"""Geometry-based maze verification helpers.

These functions work on a freshly generated, unscrolled list of ``Line`` objects
and derive connectivity purely from wall coordinates, so they do not depend on
the generator's ``sideA``/``sideB`` bookkeeping.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from ..utils.config import config
from .maze import Line

Cell = Tuple[int, int]  # (column, row)
CellGraph = Dict[Cell, Set[Cell]]


def getMazeOrigin() -> Tuple[int, int]:
    """Top-left corner of cell (0, 0) for a maze built from the default config."""
    return (config.PLAYER_START_X + config.MAZE_CELL_SIZE, config.MAZE_TOP_Y)


def buildCellGraph(
    lines: List[Line],
    width: int,
    height: int,
    origin: Optional[Tuple[int, int]] = None,
) -> CellGraph:
    """Build the cell adjacency graph implied by the wall segments.

    ``width`` and ``height`` are the values passed to ``Line.generateMaze``. The
    maze has ``2 * width`` columns and ``height - 1`` rows. Two neighbouring cells
    are connected iff no line occupies their shared edge.
    """
    width = max(width, 3)
    height = max(height, 3)
    cols = width * 2
    rows = height - 1
    cell = config.MAZE_CELL_SIZE
    originX, originY = origin if origin is not None else getMazeOrigin()

    # A vertical wall is keyed by its (x, top y); a horizontal wall by (left x, y).
    verticalWalls: Set[Tuple[int, int]] = set()
    horizontalWalls: Set[Tuple[int, int]] = set()
    for line in lines:
        if line.getIsHorizontal():
            horizontalWalls.add((line.getXStart(), line.getYStart()))
        else:
            verticalWalls.add((line.getXStart(), line.getYStart()))

    graph: CellGraph = {(c, r): set() for c in range(cols) for r in range(rows)}
    for c in range(cols):
        for r in range(rows):
            if c + 1 < cols:
                wallKey = (originX + cell * (c + 1), originY + cell * r)
                if wallKey not in verticalWalls:
                    graph[(c, r)].add((c + 1, r))
                    graph[(c + 1, r)].add((c, r))
            if r + 1 < rows:
                wallKey = (originX + cell * c, originY + cell * (r + 1))
                if wallKey not in horizontalWalls:
                    graph[(c, r)].add((c, r + 1))
                    graph[(c, r + 1)].add((c, r))
    return graph


def countEdges(graph: CellGraph) -> int:
    """Number of undirected edges in the cell graph."""
    return sum(len(neighbours) for neighbours in graph.values()) // 2


def isConnected(graph: CellGraph) -> bool:
    """Whether every cell is reachable from every other (breadth-first search)."""
    if not graph:
        return True
    start = next(iter(graph))
    seen = {start}
    queue = deque([start])
    while queue:
        for neighbour in graph[queue.popleft()]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return len(seen) == len(graph)


def isSpanningTree(graph: CellGraph) -> bool:
    """Whether the graph is connected with exactly ``cells - 1`` edges (no loops)."""
    return isConnected(graph) and countEdges(graph) == len(graph) - 1


def shortestPathLength(
    graph: CellGraph, startCell: Cell, goalColumn: int
) -> Optional[int]:
    """Number of moves from ``startCell`` to any cell in ``goalColumn``."""
    if startCell not in graph:
        return None
    distances = {startCell: 0}
    queue = deque([startCell])
    while queue:
        current = queue.popleft()
        if current[0] == goalColumn:
            return distances[current]
        for neighbour in graph[current]:
            if neighbour not in distances:
                distances[neighbour] = distances[current] + 1
                queue.append(neighbour)
    return None


def getStartCell() -> Cell:
    """Maze cell the player first enters, based on the default spawn row."""
    _, originY = getMazeOrigin()
    return (0, (config.PLAYER_START_Y - originY) // config.MAZE_CELL_SIZE)
