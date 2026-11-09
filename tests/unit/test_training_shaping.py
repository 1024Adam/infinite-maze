"""Tests for ``training.shaping`` (stdlib only: no ML libraries needed)."""

import pytest

from infinite_maze.core.simulation import Simulation
from infinite_maze.utils.config import config as game_config
from tests.unit.test_training_oracle import (
    CELL,
    GEOMETRY,
    ROUTE,
    START_ROW,
    carved_tile,
    sim_with,
)
from training.oracle import MazeGeometry, OraclePolicy, distance_field
from training.shaping import PotentialField, PotentialShaper, ShapingConfig

RIGHT = game_config.get_movement_constant("RIGHT")
# 300 ticks reach 14 columns, so the field's target column stays inside the tile.
SHORT = 300


@pytest.mark.unit
class TestPotentialField:
    def test_distance_drops_by_one_per_cell_along_the_only_route(self):
        sim = sim_with(carved_tile(ROUTE), max_steps=SHORT)
        field = PotentialField(sim, SHORT)

        spawn = field.geometry.cell_of(sim)
        path = [spawn] + [cell for cell in ROUTE if cell[0] <= field.target]
        distances = [-field.distances[cell] for cell in path]

        assert spawn == (-1, START_ROW)
        assert [a - b for a, b in zip(distances, distances[1:])] == [-1] * (
            len(path) - 1
        )

    def test_potential_is_the_negated_distance_of_the_players_cell(self):
        sim = sim_with(carved_tile(ROUTE), max_steps=SHORT)
        field = PotentialField(sim, SHORT)

        assert field.potential(sim) == -field.distances[(-1, START_ROW)]

    def test_seed_zero_prefers_leaving_the_spawn_dead_end(self):
        sim = Simulation(seed=0, start_pace=0, max_steps=3000)
        field = PotentialField(sim, 3000)
        spawn = field.geometry.cell_of(sim)

        assert spawn == (-1, START_ROW)
        into_dead_end = field.distances[(spawn[0] + 1, spawn[1])]
        along_route = field.distances[(spawn[0], spawn[1] + 1)]
        assert into_dead_end > field.distances[spawn] > along_route

    def test_field_covers_every_cell_of_its_window(self):
        sim = Simulation(seed=1, start_pace=0, max_steps=3000)
        field = PotentialField(sim, 3000)

        columns = field.target - field.low + 1
        assert len(field.distances) == columns * field.geometry.rows

    def test_a_cell_outside_the_field_raises(self):
        sim = sim_with(carved_tile(ROUTE), max_steps=SHORT)
        field = PotentialField(sim, SHORT)
        sim.player.setX(sim.player.getX() + (field.target + 3) * CELL)

        with pytest.raises(ValueError, match="outside"):
            field.potential(sim)

    def test_field_equals_a_fresh_bfs_after_scrolling_and_recycling(self):
        sim = Simulation(seed=2, start_pace=0, max_steps=3000)
        field = PotentialField(sim, 3000)
        spawn_walls = (
            set(field.geometry.verticalWalls),
            set(field.geometry.horizontalWalls),
        )

        oracle = OraclePolicy()
        oracle.reset(2)
        for _ in range(2000):  # scrolls, and recycles passed walls to the right
            action = oracle.act_privileged(sim)
            sim.step(game_config.ACTION_TO_INTENT[action])
        assert sim.getScrollOffset() > GEOMETRY.period // 2
        fresh = MazeGeometry()
        fresh.load(sim)

        assert (fresh.verticalWalls, fresh.horizontalWalls) == spawn_walls
        assert distance_field(fresh, field.low, field.target) == field.distances

    def test_oracle_behaviour_is_unchanged_by_sharing_the_cell_lookup(self):
        sim = sim_with(carved_tile(ROUTE))
        policy = OraclePolicy()

        assert policy.current_cell(sim, GEOMETRY) == GEOMETRY.cell_of(sim)


def reset_shaper(weight=44.0, gamma=0.995, seed=0, max_steps=3000):
    sim = Simulation(seed=seed, start_pace=0, max_steps=max_steps)
    shaper = PotentialShaper(ShapingConfig(weight, gamma), max_steps)
    shaper.reset(sim)
    return sim, shaper


@pytest.mark.unit
class TestPotentialShaper:
    def test_zero_weight_disables_shaping_without_building_a_field(self):
        sim, shaper = reset_shaper(weight=0.0)

        assert shaper.field is None
        assert shaper.step(sim, False) == 0.0

    def test_term_is_gamma_phi_after_minus_phi_before(self):
        sim, shaper = reset_shaper(weight=44.0, gamma=0.9)
        before = shaper.previous
        for _ in range(10):  # a few ticks into the open spawn area
            sim.step(game_config.ACTION_TO_INTENT[RIGHT])
        after = shaper.field.potential(sim)

        assert shaper.step(sim, False) == pytest.approx(44.0 * (0.9 * after - before))
        assert shaper.previous == after

    def test_terminating_transition_gets_no_shaping_term(self):
        sim, shaper = reset_shaper()

        assert shaper.step(sim, True) == 0.0

    def test_stepping_before_reset_raises(self):
        shaper = PotentialShaper(ShapingConfig(44.0, 0.995), 100)

        with pytest.raises(RuntimeError):
            shaper.step(Simulation(seed=0, max_steps=100), False)
