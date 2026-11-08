"""Functional tests for the baselines and the evaluation harness."""

import statistics

import pytest

from infinite_maze.core.replay import compareExpected, runReplay
from training import config
from training.evaluate import evaluate, record_replay
from training.policies import (
    AlwaysRightPolicy,
    HeuristicPolicy,
    RandomPolicy,
    make_policy,
)


class TestEvaluateHarness:
    def test_evaluate_is_deterministic_across_runs(self):
        seeds = [0, 1, 2]

        first = evaluate(HeuristicPolicy(), seeds, max_steps=200)
        second = evaluate(HeuristicPolicy(), seeds, max_steps=200)

        assert first == second
        assert [e.seed for e in first.episodes] == seeds

    def test_every_episode_ends_terminated_or_truncated(self):
        report = evaluate(AlwaysRightPolicy(), [0, 1], max_steps=150)

        assert all(e.terminated != e.truncated for e in report.episodes)
        assert all(e.steps <= 150 for e in report.episodes)

    def test_random_policy_report_is_reproducible(self):
        first = evaluate(RandomPolicy(), [3, 4], max_steps=200)
        second = evaluate(RandomPolicy(), [3, 4], max_steps=200)

        assert first == second

    def test_start_pace_makes_idle_agents_die_sooner(self):
        class Idle(AlwaysRightPolicy):
            def act(self, obs):
                return 0

        report = evaluate(Idle(), [0], max_steps=200, start_pace=1)

        assert report.episodes[0].terminated
        assert report.episodes[0].steps <= 10

    def test_worst_seed_replay_reproduces_the_episode(self):
        report = evaluate(HeuristicPolicy(), [5, 6], max_steps=150)
        worst = report.worst_seed()

        replay = record_replay(HeuristicPolicy(), worst, max_steps=150)
        result = runReplay(replay)

        assert compareExpected(replay, result) == []
        episode = next(e for e in report.episodes if e.seed == worst)
        assert result["score"] == episode.score
        assert result["ticks"] == episode.steps


@pytest.mark.slow
class TestBaselineQuality:
    def test_random_policy_never_crashes_and_always_ends(self):
        report = evaluate(RandomPolicy(), range(50), max_steps=2000)

        assert len(report.episodes) == 50
        assert all(e.terminated or e.truncated for e in report.episodes)

    def test_baseline_ordering_on_validation_seeds(self):
        # If this fails the observation or reward is suspect.
        means = {}
        for name in ("random", "right", "heuristic"):
            report = evaluate(make_policy(name), config.VALIDATION_SEEDS)
            means[name] = statistics.fmean(report.returns())

        assert means["heuristic"] > means["right"] > means["random"], means
