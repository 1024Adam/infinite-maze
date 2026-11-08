"""Verification of generated mazes using geometry-derived cell graphs."""

import random
from types import SimpleNamespace

import pytest

from infinite_maze.entities.maze import Line
from infinite_maze.entities.maze_analysis import (
    buildCellGraph,
    countEdges,
    getStartCell,
    isConnected,
    isSpanningTree,
    shortestPathLength,
)
from infinite_maze.utils.config import config

GAME = SimpleNamespace(Y_MIN=config.MAZE_TOP_Y)


def _generate(width, height, seed):
    """Generate a seeded maze of the given size."""
    return Line.generateMaze(GAME, width, height, random.Random(seed))


class TestCellGraph:
    """Connectivity and solvability of generated mazes."""
    @pytest.mark.parametrize("width,height", [(3, 3), (4, 6), (5, 12)])
    @pytest.mark.parametrize("seed", range(10))
    def test_small_mazes_are_spanning_trees(self, width, height, seed):
        graph = buildCellGraph(_generate(width, height, seed), width, height)

        assert len(graph) == (width * 2) * (height - 1)
        assert isConnected(graph)
        assert countEdges(graph) == len(graph) - 1

    @pytest.mark.parametrize("height", [21, 25, 32])
    def test_height_above_original_stride_is_still_a_tree(self, height):
        # The generator once used a fixed cell-id stride of 19, which collides
        # for height > 20 and merges unrelated cells.
        for seed in range(3):
            graph = buildCellGraph(_generate(4, height, seed), 4, height)
            assert isSpanningTree(graph)

    @pytest.mark.parametrize("seed", [0, 1, 1234])
    def test_default_maze_is_solvable_left_to_right(self, seed):
        width, height = config.MAZE_ROWS, config.MAZE_COLS
        graph = buildCellGraph(_generate(width, height, seed), width, height)

        assert isSpanningTree(graph)
        length = shortestPathLength(graph, getStartCell(), width * 2 - 1)
        assert length is not None
        assert length >= width * 2 - 1

    def test_unreachable_goal_returns_none(self):
        graph = {(0, 0): set(), (1, 0): set()}

        assert shortestPathLength(graph, (0, 0), 1) is None
        assert shortestPathLength(graph, (5, 5), 1) is None
        assert not isConnected(graph)

    def test_no_walls_gives_full_grid(self):
        graph = buildCellGraph([], 3, 3)
        cols, rows = 6, 2

        assert isConnected(graph)
        assert countEdges(graph) == cols * (rows - 1) + rows * (cols - 1)
        assert not isSpanningTree(graph)
