"""Tests for ``training.windowed`` and the model/observation size guard."""

import pytest

from infinite_maze.core.simulation import Simulation
from infinite_maze.utils.config import config as game_config
from tests.unit.test_training_oracle import ROUTE, carved_tile, sim_with
from training.evaluate import evaluate
from training.oracle import OraclePolicy
from training.windowed import WindowedPlanner


@pytest.mark.unit
class TestWindowedPlanner:
    def test_large_window_follows_the_known_route(self):
        sim = sim_with(carved_tile(ROUTE))
        policy = WindowedPlanner(grid_size=45)
        policy.reset(0)
        target = 14 * game_config.MAZE_CELL_SIZE + sim.player.getX()

        for _ in range(900):
            sim.step(game_config.ACTION_TO_INTENT[policy.act_privileged(sim)])

        assert sim.player.getX() + sim.getScrollOffset() >= target
        assert not sim.isTerminated()

    def test_only_cells_inside_the_window_are_considered(self):
        sim = Simulation(seed=0, start_pace=0, max_steps=50)
        small, large = WindowedPlanner(grid_size=3), WindowedPlanner(grid_size=45)
        geometry = OraclePolicy()._geometry(sim)
        for planner in (small, large):
            planner.sim = sim

        far = (geometry.cell_of(sim)[0] + 8, 8)

        assert not small._visible(sim, geometry, far)
        assert large._visible(sim, geometry, far)

    def test_evaluation_is_deterministic_and_privileged(self):
        first = evaluate(WindowedPlanner(15), [1, 2], max_steps=120)
        second = evaluate(WindowedPlanner(15), [1, 2], max_steps=120)

        assert first == second


@pytest.mark.unit
class TestModelObservationGuard:
    def test_model_refuses_a_mismatched_raster_size(self):
        pytest.importorskip("gymnasium")
        from training.model_policy import ModelPolicy

        class FakeSpace:
            shape = (89,)

        class FakeModel:
            observation_space = FakeSpace()

        ModelPolicy(FakeModel())
        with pytest.raises(ValueError, match="observation"):
            ModelPolicy(FakeModel(), grid_size=21)
