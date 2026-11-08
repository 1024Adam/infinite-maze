"""Tests for ``training.oracle``: hand-built maze, wall safety, reproducibility."""

import pytest

from infinite_maze.core.simulation import Simulation
from infinite_maze.entities.maze import Line
from infinite_maze.utils.config import config as game_config
from tests.fixtures.training_helpers import make_sim
from tests.unit.test_simulation import _overlapsAnyWall
from training.evaluate import evaluate
from training.oracle import MazeGeometry, OraclePolicy

CELL = game_config.MAZE_CELL_SIZE
GEOMETRY = MazeGeometry()
START_ROW = (game_config.PLAYER_START_Y - GEOMETRY.originY) // CELL

# Known route: straight, a climb of four rows and back down, then straight again.
ROUTE = (
    [(c, START_ROW) for c in range(0, 6)]
    + [(5, START_ROW - r) for r in range(1, 5)]
    + [(c, START_ROW - 4) for c in range(6, 12)]
    + [(11, START_ROW - 4 + r) for r in range(1, 5)]
    + [(c, START_ROW) for c in range(12, GEOMETRY.cols)]
)


def carved_tile(route):
    """A full wall grid with only the cell-to-cell edges of ``route`` removed."""
    open_edges = set()
    for (c1, r1), (c2, r2) in zip(route, route[1:]):
        open_edges.add(
            ((max(c1, c2), r1), "v") if r1 == r2 else ((c1, max(r1, r2)), "h")
        )
    lines = []
    for col in range(GEOMETRY.cols):
        for row in range(GEOMETRY.rows):
            x = GEOMETRY.originX + col * CELL
            y = GEOMETRY.originY + row * CELL
            if col > 0 and ((col, row), "v") not in open_edges:
                lines.append(Line((x, y), (x, y + CELL)))
            if row > 0 and ((col, row), "h") not in open_edges:
                lines.append(Line((x, y), (x + CELL, y)))
    return lines


def sim_with(lines, max_steps=None):
    sim = make_sim(lines, max_steps=max_steps)
    sim.player.setX(game_config.PLAYER_START_X)
    sim.player.setY(game_config.PLAYER_START_Y)
    return sim


def run(sim, policy, steps):
    policy.reset(0)
    for _ in range(steps):
        action = policy.act_privileged(sim)
        result = sim.step(game_config.ACTION_TO_INTENT[action])
        assert not _overlapsAnyWall(sim)
        if result.terminated:
            break


@pytest.mark.unit
class TestHandBuiltMaze:
    def test_follows_the_known_route_without_entering_a_wall(self):
        sim = sim_with(carved_tile(ROUTE))
        target = GEOMETRY.originX + 14 * CELL

        run(sim, OraclePolicy(), 700)

        assert sim.player.getX() + sim.getScrollOffset() >= target
        assert not sim.isTerminated()

    def test_route_length_is_close_to_the_shortest_path(self):
        sim = sim_with(carved_tile(ROUTE))
        policy = OraclePolicy()
        policy.reset(0)
        target = GEOMETRY.originX + 14 * CELL
        lower_bound = (14 + 2 * 4) * CELL  # columns plus the detour up and down

        steps = 0
        while sim.player.getX() + sim.getScrollOffset() < target and steps < 2000:
            sim.step(game_config.ACTION_TO_INTENT[policy.act_privileged(sim)])
            steps += 1

        assert steps <= lower_bound * 1.5


@pytest.mark.unit
class TestOnGeneratedMazes:
    def test_oracle_solves_the_first_tile_on_twenty_seeds_at_pace_zero(self):
        tile_end = GEOMETRY.originX + GEOMETRY.period
        policy = OraclePolicy()
        for seed in range(20):
            sim = Simulation(seed=seed, start_pace=0, max_steps=2500)
            policy.reset(seed)
            cleared = False
            for _ in range(2500):
                action = policy.act_privileged(sim)
                result = sim.step(game_config.ACTION_TO_INTENT[action])
                assert not _overlapsAnyWall(sim), seed
                if sim.player.getX() + sim.getScrollOffset() >= tile_end:
                    cleared = True
                    break
                if result.terminated:
                    break
            assert cleared, f"seed {seed} not cleared"

    def test_evaluation_is_reproducible(self):
        first = evaluate(OraclePolicy(), [1, 2], max_steps=300)
        second = evaluate(OraclePolicy(), [1, 2], max_steps=300)

        assert first == second
