"""Tests for ``training.expert.ladder`` on hand-made reports."""

import pytest

from training.evaluate import EpisodeResult, EvalReport
from training.expert import ladder
from training.reward import RewardConfig

LIMITS = {"r2": 0.5, "r3": 0.9, "pace": 3, "max_death_rate": 0.1}


def make_report(name, mean_return, score, pace=0, terminated=False, seeds=(1, 2)):
    episodes = [
        EpisodeResult(s, score, 100, mean_return, terminated, not terminated, pace)
        for s in seeds
    ]
    return EvalReport(name, 100, 0, RewardConfig(), episodes, 4)


def rung(model, heuristic=None, oracle=None):
    heuristic = heuristic or make_report("heuristic", 10.0, 50)
    oracle = oracle or make_report("oracle", 100.0, 1000)
    return ladder(model, heuristic, oracle, LIMITS)


@pytest.mark.unit
class TestLadder:
    def test_below_the_heuristic_is_rung_zero(self):
        assert rung(make_report("m", 5.0, 900)).rung == 0

    def test_beating_the_heuristic_is_rung_one(self):
        assert rung(make_report("m", 20.0, 100)).rung == 1

    def test_oracle_fractions_give_rungs_two_and_three(self):
        assert rung(make_report("m", 20.0, 500)).rung == 2
        assert rung(make_report("m", 20.0, 900)).rung == 3

    def test_rung_four_needs_pace_and_low_death_rate(self):
        assert rung(make_report("m", 20.0, 950, pace=3)).rung == 4
        assert rung(make_report("m", 20.0, 950, pace=2)).rung == 3
        dying = make_report("m", 20.0, 950, pace=3, terminated=True)
        assert rung(dying).rung == 3

    def test_rungs_are_cumulative(self):
        # Near-oracle score but not beating the heuristic's return: still rung 0.
        assert rung(make_report("m", 5.0, 990, pace=5)).rung == 0

    def test_reports_must_share_seeds_and_settings(self):
        other_seeds = make_report("heuristic", 10.0, 50, seeds=(7, 8))

        with pytest.raises(ValueError, match="identical"):
            rung(make_report("m", 20.0, 100), heuristic=other_seeds)
