"""Unit tests for ``training.reward``, including the reward-hacking guards."""

import json

import pytest

from infinite_maze.core.simulation import StepResult
from tests.fixtures.training_helpers import IDLE, LEFT, RIGHT, make_sim, vertical
from training import config
from training.reward import (
    RewardConfig,
    RewardFunction,
    load_reward_config,
)


def result(moved_x=0.0, terminated=False, truncated=False):
    """A StepResult with the given horizontal displacement."""
    return StepResult(0, (moved_x, 0), terminated, truncated)


def total_reward(sim, intents, reward=None):
    """Sum of step rewards for ``intents``."""
    reward_fn = RewardFunction(reward)
    return sum(reward_fn(sim.step(intent)).total for intent in intents)


class TestBreakdown:
    def test_progress_uses_horizontal_displacement(self):
        b = RewardFunction()(result(moved_x=1))

        assert b.progress == 1.0
        assert b.step == -config.REWARD_STEP_PENALTY
        assert b.terminal == 0.0
        assert b.total == pytest.approx(1.0 - config.REWARD_STEP_PENALTY)

    def test_leftward_movement_is_negative_progress(self):
        assert RewardFunction()(result(moved_x=-1)).progress == -1.0

    def test_termination_applies_the_death_penalty(self):
        b = RewardFunction()(result(terminated=True))

        assert b.terminal == -config.REWARD_DEATH_PENALTY
        assert b.total == pytest.approx(b.step + b.terminal)

    def test_truncation_bonus(self):
        reward = RewardFunction(RewardConfig(truncation_bonus=5.0))

        assert reward(result(truncated=True)).terminal == 5.0
        assert reward(result()).terminal == 0.0

    def test_weights_are_configurable(self):
        reward = RewardFunction(RewardConfig(progress_weight=2.0, step_penalty=0.5))
        b = reward(result(moved_x=1))

        assert (b.progress, b.step, b.total) == (2.0, -0.5, 1.5)

    def test_total_is_the_sum_of_parts(self):
        b = RewardFunction()(result(moved_x=1, terminated=True))

        assert b.total == pytest.approx(b.progress + b.step + b.terminal)


class TestRewardHackingGuards:
    def test_oscillating_scores_below_always_right(self):
        oscillating = make_sim()
        right = make_sim()

        wiggle = total_reward(oscillating, [RIGHT, LEFT] * 50)
        push = total_reward(right, [RIGHT] * 100)

        assert wiggle == pytest.approx(-100 * config.REWARD_STEP_PENALTY)
        assert wiggle < 0 < push

    def test_wall_hugging_only_accrues_the_step_penalty(self):
        sim = make_sim([vertical(211, 100, 300)])

        total = total_reward(sim, [RIGHT] * 50)

        assert total == pytest.approx(-50 * config.REWARD_STEP_PENALTY)

    def test_score_floor_cannot_be_exploited(self):
        # In-game, moving left at score 0 costs nothing; here it must cost.
        sim = make_sim()
        assert sim.game.getScore() == 0

        step = sim.step(LEFT)
        b = RewardFunction()(step)

        assert step.score_delta == 0
        assert b.progress == -1.0
        assert b.total < -config.REWARD_STEP_PENALTY

    def test_dying_is_worse_than_idling_through_the_whole_episode(self):
        max_steps = config.EVAL_MAX_STEPS
        survive = make_sim()
        survive_total = total_reward(survive, [IDLE] * max_steps)

        dying = make_sim(start_pace=1)
        dying.player.setX(dying.game.X_MIN)
        die_total = 0.0
        reward_fn = RewardFunction()
        while True:
            step = dying.step(IDLE)
            die_total += reward_fn(step).total
            if step.terminated:
                break

        assert die_total < survive_total

    def test_default_death_penalty_covers_the_configured_episode_lengths(self):
        RewardConfig().validate(config.EVAL_MAX_STEPS)
        RewardConfig().validate(5000)
        RewardConfig().validate(None)

    def test_validate_rejects_a_death_penalty_that_is_too_small(self):
        with pytest.raises(ValueError, match="death_penalty"):
            RewardConfig(death_penalty=5.0, step_penalty=0.01).validate(1000)

    def test_validate_rejects_equality(self):
        with pytest.raises(ValueError):
            RewardConfig(death_penalty=10.0, step_penalty=0.01).validate(1000)


class TestRewardConfigFile:
    def test_loads_partial_overrides(self, tmp_path):
        path = tmp_path / "reward.json"
        path.write_text(json.dumps({"step_penalty": 0.5}))

        loaded = load_reward_config(str(path))

        assert loaded.step_penalty == 0.5
        assert loaded.progress_weight == config.REWARD_PROGRESS_WEIGHT

    def test_unknown_keys_are_rejected(self, tmp_path):
        path = tmp_path / "reward.json"
        path.write_text(json.dumps({"progres_weight": 1}))

        with pytest.raises(ValueError, match="unknown"):
            load_reward_config(str(path))

    def test_non_object_is_rejected(self, tmp_path):
        path = tmp_path / "reward.json"
        path.write_text("[1, 2]")

        with pytest.raises(ValueError, match="object"):
            load_reward_config(str(path))
