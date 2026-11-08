"""Unit tests for ``training.policies``."""

import random

import pytest

from training import config
from training.observation import Observation
from training.policies import (
    ACTIONS,
    DO_NOTHING,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AlwaysRightPolicy,
    HeuristicPolicy,
    Policy,
    RandomPolicy,
    RasterHeuristicPolicy,
    make_policy,
)

OPEN = 0.5
BLOCKED = 0.0


def obs(right=OPEN, left=OPEN, up=OPEN, down=OPEN, grid=None):
    """Observation with the given ray fractions."""
    size = config.OBS_GRID_SIZE
    return Observation(
        grid=grid or [[0] * size for _ in range(size)],
        rays=(right, left, up, down),
        scalars=(0.0, 0.0, 0.0, 0.0),
    )


class TestBasePolicies:
    def test_policy_is_abstract(self):
        with pytest.raises(TypeError):
            Policy()

    def test_random_policy_returns_valid_actions(self):
        policy = RandomPolicy()
        policy.reset(1)

        assert {policy.act(obs()) for _ in range(200)} <= set(ACTIONS)

    def test_random_policy_is_reproducible_per_seed(self):
        first, second = RandomPolicy(), RandomPolicy(random.Random(99))

        first.reset(7)
        second.reset(7)

        assert [first.act(obs()) for _ in range(50)] == [
            second.act(obs()) for _ in range(50)
        ]

    def test_random_policy_differs_between_seeds(self):
        policy = RandomPolicy()
        policy.reset(1)
        first = [policy.act(obs()) for _ in range(50)]
        policy.reset(2)
        second = [policy.act(obs()) for _ in range(50)]

        assert first != second

    def test_always_right(self):
        assert AlwaysRightPolicy().act(obs(right=BLOCKED)) == RIGHT

    def test_make_policy(self):
        assert make_policy("random").name == "random"
        assert make_policy("heuristic").name == "heuristic"
        with pytest.raises(ValueError, match="unknown policy"):
            make_policy("nope")


class TestHeuristicPolicy:
    def test_goes_right_when_open(self):
        assert HeuristicPolicy().act(obs()) == RIGHT

    def test_blocked_ahead_goes_toward_the_longer_side(self):
        assert HeuristicPolicy().act(obs(right=BLOCKED, up=0.9, down=0.2)) == UP
        assert HeuristicPolicy().act(obs(right=BLOCKED, up=0.2, down=0.9)) == DOWN

    def test_only_one_vertical_side_open(self):
        assert HeuristicPolicy().act(obs(right=BLOCKED, up=BLOCKED, down=0.3)) == DOWN

    def test_dead_end_backs_up_or_waits(self):
        dead_end = obs(right=BLOCKED, up=BLOCKED, down=BLOCKED)

        assert HeuristicPolicy().act(dead_end) == LEFT
        assert HeuristicPolicy().act(obs(BLOCKED, BLOCKED, BLOCKED, BLOCKED)) == (
            DO_NOTHING
        )

    def test_symmetric_corridor_does_not_flip_every_step(self):
        policy = HeuristicPolicy()
        corridor = obs(right=BLOCKED, up=OPEN, down=OPEN)

        actions = [policy.act(corridor) for _ in range(100)]

        flips = sum(a != b for a, b in zip(actions, actions[1:]))
        assert flips == 0
        assert set(actions) == {UP}

    def test_commits_to_a_direction_while_it_stays_open(self):
        policy = HeuristicPolicy(commit_steps=5)
        assert policy.act(obs(right=BLOCKED, up=0.9, down=0.1)) == UP

        # Down now has more room, but the policy keeps going up.
        for _ in range(4):
            assert policy.act(obs(right=BLOCKED, up=0.3, down=0.9)) == UP

        # After the commitment it re-evaluates.
        assert policy.act(obs(right=BLOCKED, up=0.3, down=0.9)) == DOWN

    def test_commitment_ends_when_the_direction_is_blocked(self):
        policy = HeuristicPolicy()
        policy.act(obs(right=BLOCKED, up=0.9, down=0.1))

        assert policy.act(obs(right=BLOCKED, up=BLOCKED, down=0.4)) == DOWN

    def test_opening_to_the_right_wins_and_clears_commitment(self):
        policy = HeuristicPolicy()
        policy.act(obs(right=BLOCKED, up=0.9, down=0.1))

        assert policy.act(obs(right=OPEN)) == RIGHT
        assert policy.direction is None

    def test_reset_clears_state(self):
        policy = HeuristicPolicy()
        policy.act(obs(right=BLOCKED, up=0.9, down=0.1))
        policy.reset(3)

        assert policy.direction is None and policy.remaining == 0

    def test_wall_one_pixel_ahead_counts_as_blocked(self):
        one_px = 1 / config.OBS_RAY_CAP_PX

        assert HeuristicPolicy().act(obs(right=one_px, up=0.5, down=0.1)) == UP


class TestRasterHeuristicPolicy:
    def grid(self, walls):
        size = config.OBS_GRID_SIZE
        grid = [[0] * size for _ in range(size)]
        for row, col in walls:
            grid[row][col] = 1
        return grid

    def test_open_ahead_goes_right(self):
        assert RasterHeuristicPolicy().act(obs(grid=self.grid([]))) == RIGHT

    def test_wall_ahead_goes_toward_more_free_cells(self):
        center = config.OBS_GRID_SIZE // 2
        grid = self.grid([(center, center + 1), (center - 2, center)])

        # Up has 1 free cell, down has 4.
        assert RasterHeuristicPolicy().act(obs(grid=grid)) == DOWN
