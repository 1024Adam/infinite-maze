"""Unit tests for ``training.observation``."""

import pytest

from infinite_maze.core.simulation import Simulation
from infinite_maze.utils.config import config as game_config
from tests.fixtures.training_helpers import (
    DOWN,
    IDLE,
    LEFT,
    RIGHT,
    UP,
    horizontal,
    make_sim,
    vertical,
)
from training import config
from training.observation import (
    build_observation,
    flatten_observation,
    observation_size,
    window_origin,
)

CAP = config.OBS_RAY_CAP_PX
CELL = config.OBS_GRID_CELL_PX
SIZE = config.OBS_GRID_SIZE


def cells(obs):
    """Set of (row, col) raster cells that contain a wall."""
    return {(r, c) for r, row in enumerate(obs.grid) for c, v in enumerate(row) if v}


class TestEmptyMaze:
    """Player box is [200, 210] x [200, 210]; the window origin is (156, 156)."""

    def test_empty_maze_has_empty_grid_and_full_rays(self):
        obs = build_observation(make_sim())

        assert len(obs.grid) == SIZE and all(len(r) == SIZE for r in obs.grid)
        assert cells(obs) == set()
        assert obs.rays == (1.0, 1.0, 1.0, 1.0)

    def test_window_is_centred_on_the_player(self):
        sim = make_sim()

        assert window_origin(sim) == (156, 156)
        assert (205 - 156) // CELL == SIZE // 2

    def test_scalars_are_normalised(self):
        obs = build_observation(make_sim())

        assert len(obs.scalars) == 4
        assert all(0.0 <= v <= 1.0 for v in obs.scalars)


class TestRays:
    def test_wall_directly_right(self):
        obs = build_observation(make_sim([vertical(215, 190, 230)]))

        assert obs.rays == (pytest.approx(5 / CAP), 1.0, 1.0, 1.0)

    def test_wall_directly_left(self):
        obs = build_observation(make_sim([vertical(190, 190, 230)]))

        assert obs.rays == (1.0, pytest.approx(10 / CAP), 1.0, 1.0)

    def test_wall_directly_above(self):
        obs = build_observation(make_sim([horizontal(193, 190, 230)]))

        assert obs.rays == (1.0, 1.0, pytest.approx(7 / CAP), 1.0)

    def test_wall_directly_below(self):
        obs = build_observation(make_sim([horizontal(218, 190, 230)]))

        assert obs.rays == (1.0, 1.0, 1.0, pytest.approx(8 / CAP))

    def test_distance_is_capped(self):
        far = 210 + CAP + 40
        obs = build_observation(make_sim([vertical(far, 190, 230)]))

        assert obs.rays[0] == 1.0

    def test_wall_beside_the_player_does_not_block_a_ray(self):
        obs = build_observation(make_sim([vertical(215, 100, 190)]))

        assert obs.rays[0] == 1.0

    def test_horizontal_wall_touching_the_box_blocks_like_the_engine(self):
        # The engine blocks rightward movement when a horizontal wall touching
        # the player's edge starts one pixel ahead.
        obs = build_observation(make_sim([horizontal(210, 213, 235)]))

        assert obs.rays[0] == pytest.approx(3 / CAP)

    def test_nearest_wall_wins(self):
        lines = [vertical(260, 190, 230), vertical(222, 190, 230)]

        assert build_observation(make_sim(lines)).rays[0] == pytest.approx(12 / CAP)

    def test_blocked_implies_ray_within_one_step_on_a_real_maze(self):
        # Cross-check against the engine's own collision test.
        sim = Simulation(seed=1, max_steps=None)
        intents = [RIGHT, RIGHT, RIGHT, UP, DOWN, RIGHT, LEFT, IDLE]
        directions = {RIGHT: 0, LEFT: 1, UP: 2, DOWN: 3}
        from infinite_maze.core import simulation as sim_module

        codes = {
            RIGHT: sim_module.RIGHT,
            LEFT: sim_module.LEFT,
            UP: sim_module.UP,
            DOWN: sim_module.DOWN,
        }
        for step in range(150):
            sim.step(intents[(step * 7 + step // 3) % len(intents)])
            rays = build_observation(sim).rays
            for direction, index in directions.items():
                if sim._isBlocked(codes[direction]):
                    assert rays[index] * CAP <= sim.player.getSpeed()


class TestRaster:
    def test_horizontal_wall_marks_every_cell_it_crosses(self):
        # y = 163 is row 0 (156..166); x 160..192 crosses columns 0..3.
        obs = build_observation(make_sim([horizontal(163, 160, 192)]))

        assert cells(obs) == {(0, 0), (0, 1), (0, 2), (0, 3)}

    def test_vertical_wall_marks_every_cell_it_crosses(self):
        # x = 205 is column 4; y 170..201 crosses rows 1..4.
        obs = build_observation(make_sim([vertical(205, 170, 201)]))

        assert cells(obs) == {(1, 4), (2, 4), (3, 4), (4, 4)}

    def test_segment_ending_on_a_cell_boundary_does_not_mark_the_next_cell(self):
        # x 156..178 covers exactly columns 0 and 1.
        obs = build_observation(make_sim([horizontal(160, 156, 178)]))

        assert cells(obs) == {(0, 0), (0, 1)}

    def test_wall_straddling_the_window_edge_is_clipped(self):
        # Starts far left of the window and ends inside column 1.
        obs = build_observation(make_sim([horizontal(160, 100, 170)]))

        assert cells(obs) == {(0, 0), (0, 1)}

    def test_wall_outside_the_window_marks_nothing(self):
        lines = [horizontal(300, 160, 190), vertical(400, 150, 250)]

        assert cells(build_observation(make_sim(lines))) == set()

    def test_raster_ignores_scroll_offsets(self):
        lines = [horizontal(163, 160, 192), vertical(205, 170, 201)]
        reference = build_observation(make_sim(lines))

        shifted_lines = [horizontal(163, 163, 195), vertical(208, 170, 201)]
        shifted = build_observation(make_sim(shifted_lines, x=203))

        assert shifted.grid == reference.grid
        assert shifted.rays == reference.rays

    def test_window_follows_the_player(self):
        wall = [horizontal(263, 260, 292)]

        near = build_observation(make_sim(wall, x=200, y=200))
        centred = build_observation(make_sim(wall, x=260, y=260))

        assert cells(near) == set()
        assert cells(centred) != set()


class TestConfigurableWindow:
    def test_larger_window_and_ray_cap(self):
        sim = make_sim([vertical(300, 150, 250)])

        obs = build_observation(sim, grid_size=13, cell_px=11, ray_cap_px=200)

        assert len(obs.grid) == 13
        assert obs.rays[0] == pytest.approx(90 / 200)


class TestPacePhase:
    def test_phase_does_not_depend_on_max_steps(self):
        short = make_sim(max_steps=100)
        unlimited = make_sim(max_steps=None)
        for sim in (short, unlimited):
            for _ in range(50):
                sim.step(IDLE)

        assert build_observation(short).scalars == build_observation(unlimited).scalars

    def test_phase_wraps_at_the_pace_interval(self):
        sim = make_sim(max_steps=None)
        for _ in range(game_config.PACE_UPDATE_TICKS // 2):
            sim.step(IDLE)

        assert build_observation(sim).scalars[3] == pytest.approx(0.5)


class TestFlatten:
    def test_layout_is_grid_then_rays_then_scalars(self):
        obs = build_observation(make_sim([vertical(215, 190, 230)]))

        flat = flatten_observation(obs)

        assert len(flat) == observation_size()
        assert flat[: SIZE * SIZE] == [float(c) for row in obs.grid for c in row]
        assert flat[SIZE * SIZE : SIZE * SIZE + 4] == list(obs.rays)
        assert flat[-4:] == list(obs.scalars)

    def test_values_are_floats_in_unit_range(self):
        flat = flatten_observation(build_observation(make_sim()))

        assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in flat)
