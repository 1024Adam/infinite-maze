"""Reward-ordering gate for oracle shaping (stdlib only, plan 4 WP3)."""

import pytest

from training import config
from training.shaping import ShapingConfig
from training.shaping_check import (
    EXPECTED_ORDER,
    mean_training_returns,
    ordering_holds,
)

GATE_SEEDS = list(config.TRAIN_SEEDS)[:20]


@pytest.mark.slow
@pytest.mark.functional
class TestRewardOrderingGate:
    def test_oracle_beats_heuristic_beats_right_beats_random_with_shaping_on(self):
        results = mean_training_returns(
            GATE_SEEDS, ShapingConfig(weight=44.0, gamma=0.995), reward_scale=0.05
        )

        ranked = [results[name].discounted for name in EXPECTED_ORDER]
        assert ordering_holds(results), dict(zip(EXPECTED_ORDER, ranked))
        assert ranked[0] - ranked[1] > 10.0  # the oracle clearly leads
