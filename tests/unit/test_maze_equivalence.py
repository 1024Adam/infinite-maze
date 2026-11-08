"""``Line.generateMaze`` must match the frozen legacy algorithm exactly."""

import random

import pytest

from infinite_maze.core.game import Game
from infinite_maze.entities.maze import Line
from infinite_maze.utils.config import config
from tests.fixtures.legacy_maze import legacyGenerateMaze


def snapshot(lines):
    return [(l.getStart(), l.getEnd(), l.getSideA(), l.getSideB()) for l in lines]


def generate(fn, width, height, seed):
    rng = random.Random(seed)
    lines = fn(Game(headless=True), width, height, rng)
    return snapshot(lines), rng.random()


class TestLegacyEquivalence:
    @pytest.mark.parametrize("width,height", [(3, 3), (4, 6), (5, 12)])
    @pytest.mark.parametrize("seed", range(200))
    def test_small_mazes_match_the_legacy_algorithm(self, width, height, seed):
        # The trailing rng draw also proves the same number of draws was used.
        assert generate(Line.generateMaze, width, height, seed) == generate(
            legacyGenerateMaze, width, height, seed
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_default_size_mazes_match_the_legacy_algorithm(self, seed):
        size = (config.MAZE_ROWS, config.MAZE_COLS)

        assert generate(Line.generateMaze, *size, seed) == generate(
            legacyGenerateMaze, *size, seed
        )
