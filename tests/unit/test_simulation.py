"""Unit tests for the rules implementation in ``Simulation``."""

import random
from unittest.mock import patch

import pytest

from infinite_maze.core.simulation import Simulation
from infinite_maze.entities.maze import Line
from infinite_maze.utils.config import config

RIGHT, LEFT, UP, DOWN = (1, 0), (-1, 0), (0, -1), (0, 1)
IDLE = (0, 0)


def makeSim(lines=None, **kwargs):
    """Simulation with a hand-made maze (generation is skipped for speed)."""
    with patch.object(Line, "generateMaze", return_value=[]):
        sim = Simulation(**kwargs)
    sim.lines = list(lines or [])
    return sim


def place(sim, x, y):
    """Put the player at (x, y)."""
    sim.player.setX(x)
    sim.player.setY(y)


def vertical(x, yStart, yEnd):
    """A vertical wall segment at ``x``."""
    return Line((x, yStart), (x, yEnd))


def horizontal(y, xStart, xEnd):
    """A horizontal wall segment at ``y``."""
    return Line((xStart, y), (xEnd, y))


class TestBlocking:
    """Player is 10x10, speed 1, standing at (200, 200)."""

    @pytest.mark.parametrize(
        "intent,wall",
        [
            (RIGHT, vertical(211, 190, 230)),
            (RIGHT, horizontal(205, 211, 233)),
            (LEFT, vertical(199, 190, 230)),
            (LEFT, horizontal(205, 177, 199)),
            (DOWN, horizontal(211, 190, 230)),
            (DOWN, vertical(205, 211, 233)),
            (UP, horizontal(199, 190, 230)),
            (UP, vertical(205, 177, 199)),
        ],
    )
    def test_wall_in_front_blocks_movement(self, intent, wall):
        sim = makeSim([wall])
        place(sim, 200, 200)
        sim.game.setScore(5)

        result = sim.step(intent)

        assert result.moved == (0, 0)
        assert result.score_delta == 0
        assert sim.player.getPosition() == (200, 200)

    @pytest.mark.parametrize("intent", [RIGHT, LEFT, UP, DOWN])
    def test_free_movement_moves_one_step(self, intent):
        sim = makeSim([vertical(260, 100, 300)])
        place(sim, 200, 200)
        sim.game.setScore(5)

        result = sim.step(intent)

        assert result.moved == intent
        assert sim.player.getPosition() == (200 + intent[0], 200 + intent[1])

    def test_wall_beside_does_not_block(self):
        sim = makeSim([vertical(211, 100, 190)])
        place(sim, 200, 200)

        assert sim.step(RIGHT).moved == (1, 0)

    def test_diagonal_slides_along_wall(self):
        sim = makeSim([vertical(211, 100, 400)])
        place(sim, 200, 200)

        result = sim.step((1, 1))

        assert result.moved == (0, 1)


class TestScoring:
    """Score changes from horizontal movement, with a floor at zero."""
    def test_right_scores_one(self):
        sim = makeSim()
        place(sim, 200, 200)

        assert sim.step(RIGHT).score_delta == 1
        assert sim.game.getScore() == 1

    def test_left_decrements_but_floors_at_zero(self):
        sim = makeSim()
        place(sim, 200, 200)
        sim.game.setScore(1)

        assert sim.step(LEFT).score_delta == -1
        assert sim.step(LEFT).score_delta == 0
        assert sim.game.getScore() == 0

    def test_vertical_movement_does_not_score(self):
        sim = makeSim()
        place(sim, 200, 200)

        assert sim.step(DOWN).score_delta == 0


class TestScrollingAndBounds:
    """Scrolling, death, screen clamps and line recycling."""
    def test_scroll_happens_every_scroll_interval_by_pace(self):
        wall = vertical(300, 100, 300)
        sim = makeSim([wall], start_pace=2)
        place(sim, 200, 200)

        for _ in range(config.SCROLL_INTERVAL_TICKS - 1):
            sim.step(IDLE)
        assert sim.player.getX() == 200
        assert wall.getXStart() == 300

        sim.step(IDLE)
        assert sim.player.getX() == 198
        assert wall.getXStart() == 298

    def test_pace_zero_never_scrolls(self):
        wall = vertical(300, 100, 300)
        sim = makeSim([wall])
        place(sim, 200, 200)

        for _ in range(config.SCROLL_INTERVAL_TICKS * 3):
            sim.step(IDLE)

        assert wall.getXStart() == 300

    def test_pushed_past_x_min_terminates(self):
        sim = makeSim(start_pace=1)
        place(sim, config.PLAYER_START_X, 200)

        for _ in range(config.SCROLL_INTERVAL_TICKS - 1):
            assert not sim.step(IDLE).terminated
        result = sim.step(IDLE)

        assert result.terminated
        assert sim.isTerminated()
        assert not sim.game.isActive()

    def test_walking_left_off_the_start_terminates(self):
        sim = makeSim()
        place(sim, config.PLAYER_START_X, 200)

        assert sim.step(LEFT).terminated

    def test_x_max_clamps_player_and_scrolls_world(self):
        wall = vertical(500, 100, 300)
        sim = makeSim([wall])
        place(sim, sim.game.X_MAX, 200)

        sim.step(RIGHT)

        assert sim.player.getX() == sim.game.X_MAX
        assert wall.getXStart() == 500 - sim.player.getSpeed()

    def test_y_is_clamped(self):
        sim = makeSim()
        place(sim, 200, sim.game.Y_MIN)
        sim.step(UP)
        assert sim.player.getY() == sim.game.Y_MIN

        place(sim, 200, sim.game.Y_MAX)
        sim.step(DOWN)
        assert sim.player.getY() == sim.game.Y_MAX

    def test_passed_lines_are_recycled_to_the_far_edge(self):
        far = horizontal(100, 400, 400 + config.MAZE_CELL_SIZE)
        passed = horizontal(120, config.PLAYER_START_X - 1, 100)
        passedWall = vertical(config.PLAYER_START_X - 1, 100, 122)
        sim = makeSim([far, passed, passedWall])
        place(sim, 200, 200)

        sim.step(IDLE)

        xMax = 400 + config.MAZE_CELL_SIZE
        assert (passed.getXStart(), passed.getXEnd()) == (
            xMax,
            xMax + config.MAZE_CELL_SIZE,
        )
        assert (passedWall.getXStart(), passedWall.getXEnd()) == (xMax, xMax)


class TestTimeAndEpisodes:
    """Tick-based pace, episode caps, reset and seed handling."""
    def test_pace_increments_exactly_at_pace_update_ticks(self):
        sim = makeSim()
        place(sim, 200, 200)

        for _ in range(config.PACE_UPDATE_TICKS - 1):
            sim.step(IDLE)
        assert sim.game.getPace() == 0

        sim.step(IDLE)
        assert sim.game.getPace() == 1
        assert sim.getTicks() == config.PACE_UPDATE_TICKS

    def test_start_pace_is_honoured_and_restored_on_reset(self):
        sim = makeSim(start_pace=3)
        assert sim.game.getPace() == 3

        sim.game.setPace(9)
        with patch.object(Line, "generateMaze", return_value=[]):
            sim.reset()

        assert sim.game.getPace() == 3

    def test_truncated_at_max_steps(self):
        sim = makeSim(max_steps=3)
        place(sim, 200, 200)

        flags = [sim.step(IDLE).truncated for _ in range(3)]

        assert flags == [False, False, True]

    def test_no_truncation_by_default(self):
        sim = makeSim()
        place(sim, 200, 200)

        assert not any(sim.step(IDLE).truncated for _ in range(50))

    def test_stalled_agent_is_ended_by_truncation(self):
        sim = makeSim([vertical(211, 0, 500)], max_steps=20)
        place(sim, 200, 200)

        results = [sim.step(RIGHT) for _ in range(20)]

        assert results[-1].truncated
        assert sim.game.getScore() == 0

    def test_reset_clears_episode_state(self):
        sim = makeSim()
        place(sim, 200, 200)
        sim.step(RIGHT)
        with patch.object(Line, "generateMaze", return_value=[]):
            sim.reset()

        assert sim.game.getScore() == 0
        assert sim.getTicks() == 0
        assert sim.getSteps() == 0
        assert not sim.isTerminated()
        assert sim.player.getPosition() == (
            config.PLAYER_START_X,
            config.PLAYER_START_Y,
        )

    def test_explicit_seed_is_reused_on_reset(self):
        sim = makeSim(seed=77)
        with patch.object(Line, "generateMaze", return_value=[]):
            sim.reset()
            assert sim.getSeed() == 77
            sim.reset(seed=5)
            assert sim.getSeed() == 5
            sim.reset()
            assert sim.getSeed() == 5
        assert sim.game.getSeed() == 5

    def test_unspecified_seed_changes_every_episode(self):
        with patch("infinite_maze.core.simulation.newSeed", side_effect=[11, 22, 33]):
            sim = makeSim()
            with patch.object(Line, "generateMaze", return_value=[]):
                sim.reset()
                sim.reset()
        assert sim.getSeed() == 33


class TestQueries:
    """Read-only helpers used by controllers and replays."""
    def test_closest_line_is_nearest_start_ahead_of_player(self):
        sim = makeSim([vertical(300, 0, 10), vertical(250, 0, 10), vertical(100, 0, 10)])
        place(sim, 200, 200)

        assert sim.getClosestLineX() == 250

    def test_closest_line_uses_sentinel_when_nothing_ahead(self):
        sim = makeSim([vertical(100, 0, 10)])
        place(sim, 200, 200)

        assert sim.getClosestLineX() == config.NO_LINE_SENTINEL_X

    def test_state_snapshot_is_json_friendly(self):
        sim = makeSim()
        place(sim, 200.0, 200)
        state = sim.getState()

        assert state == {
            "score": 0,
            "ticks": 0,
            "pace": 0,
            "player": [200, 200],
            "terminated": False,
        }
        assert isinstance(state["player"][0], int)


def fuzzIntents(seed, count):
    """Deterministic pseudo-random intents biased towards moving right."""
    rng = random.Random(seed)
    options = [RIGHT, RIGHT, RIGHT, UP, DOWN, (1, 1), (1, -1), LEFT, IDLE]
    return [rng.choice(options) for _ in range(count)]


class TestDeterminismAndInvariants:
    """Same inputs give the same game; the world is periodic and walls are solid."""
    def test_same_seed_and_intents_give_identical_state(self):
        intents = fuzzIntents(3, 5000)
        first = Simulation(seed=21, max_steps=None)
        second = Simulation(seed=21, max_steps=None)

        for intent in intents:
            a = first.step(intent)
            b = second.step(intent)
            assert a == b

        assert first.getState() == second.getState()
        assert [(l.getStart(), l.getEnd()) for l in first.getLines()] == [
            (l.getStart(), l.getEnd()) for l in second.getLines()
        ]

    def test_world_is_periodic_with_period_of_one_maze_width(self):
        period = 2 * config.MAZE_ROWS * config.MAZE_CELL_SIZE

        def pattern(sim):
            # Position of each wall modulo the period, relative to the left edge.
            return sorted(
                (
                    line.getYStart(),
                    (line.getXStart() - config.PLAYER_START_X) % period,
                    line.getXEnd() - line.getXStart(),
                    line.getYEnd(),
                )
                for line in sim.getLines()
            )

        sim = Simulation(seed=4, max_steps=None)
        before = pattern(sim)

        for _ in range(3 * period // config.MAZE_CELL_SIZE):
            sim._scrollLines(config.MAZE_CELL_SIZE)
            sim._recycleLines()

        assert pattern(sim) == before

    @pytest.mark.parametrize("pace", [0, 1, 2, 3])
    def test_player_never_enters_a_wall(self, pace):
        # Scrolling moves the player and the walls by the same amount, so a
        # higher pace cannot make the player tunnel through a wall.
        sim = Simulation(seed=8, start_pace=pace, max_steps=None)
        for intent in fuzzIntents(pace, 500):
            sim.step(intent)
            if sim.isTerminated():
                break
            assert not _overlapsAnyWall(sim)


def _overlapsAnyWall(sim):
    """Whether the player's box strictly overlaps the interior of any wall."""
    px, py = sim.player.getPosition()
    pw, ph = sim.player.getWidth(), sim.player.getHeight()
    for line in sim.getLines():
        xs, ys, xe, ye = (
            line.getXStart(),
            line.getYStart(),
            line.getXEnd(),
            line.getYEnd(),
        )
        if line.getIsHorizontal():
            if py < ys < py + ph and px < xe and px + pw > xs:
                return True
        elif px < xs < px + pw and py < ye and py + ph > ys:
            return True
    return False


class TestSeedVisibility:
    """The HUD only shows explicitly requested seeds."""
    def test_only_explicit_seeds_are_visible(self):
        assert makeSim(seed=5).game.isSeedVisible()
        assert not makeSim().game.isSeedVisible()

    def test_reset_with_explicit_seed_makes_seed_visible(self):
        sim = makeSim()
        with patch.object(Line, "generateMaze", return_value=[]):
            sim.reset(seed=9)
        assert sim.game.isSeedVisible()
