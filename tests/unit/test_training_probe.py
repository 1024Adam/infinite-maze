"""Tests for ``training.probe`` (skipped without the ``ml-train`` group)."""

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

import numpy as np  # noqa: E402

from infinite_maze.entities.maze import Line  # noqa: E402
from tests.fixtures.training_helpers import make_sim  # noqa: E402
from training import config  # noqa: E402
from training.evaluate import evaluate  # noqa: E402
from training.experiment import ExperimentConfig  # noqa: E402
from training.observation import build_observation  # noqa: E402
from training.probe import (  # noqa: E402
    ProbeConfig,
    Rollout,
    check_probe_seeds,
    collect,
    fit,
    g1_reading,
    run_probe,
)

SMALL = ProbeConfig(grid_size=9, epochs=2, max_steps=60, betas=(1.0, 0.0), hidden=(8,))


@pytest.mark.unit
class TestSeedSafety:
    def test_validation_and_test_seeds_are_refused_for_labelling(self):
        with pytest.raises(ValueError):
            check_probe_seeds([1000], [0])
        with pytest.raises(ValueError):
            check_probe_seeds([3000], [2000])

    def test_overlapping_splits_are_refused(self):
        with pytest.raises(ValueError, match="overlap"):
            check_probe_seeds([3000, 0], [0])

    def test_default_splits_are_disjoint_training_seeds(self):
        check_probe_seeds(config.PROBE_FIT_SEEDS, config.PROBE_HOLDOUT_SEEDS)


@pytest.mark.unit
class TestObservationIsNotPrivileged:
    def test_walls_beyond_window_and_rays_do_not_change_the_observation(self):
        near = [Line((260, 200), (260, 222))]
        far = near + [Line((560, 100), (560, 122)), Line((600, 300), (700, 300))]

        a = build_observation(make_sim(near, x=200, y=200))
        b = build_observation(make_sim(far, x=200, y=200))

        assert a == b

    def test_experiment_config_has_no_privileged_switch(self):
        names = set(ExperimentConfig.__dataclass_fields__)

        assert not {n for n in names if "oracle" in n or "privileged" in n}


@pytest.mark.unit
class TestRolloutsAndFit:
    def test_oracle_rollout_labels_the_states_it_visits(self):
        batch = collect(Rollout(3000, SMALL, None, 1.0, "sample", 0))

        assert batch.x.shape[1] == SMALL.input_size()
        assert len(batch.y) == len(batch.decision) > 0
        assert set(np.unique(batch.y)) <= {0, 1, 2, 3, 4}

    def test_raster_only_input_drops_rays_and_scalars(self):
        cfg = ProbeConfig(grid_size=9, extras=False, max_steps=40)

        batch = collect(Rollout(3000, cfg, None, 1.0, "sample", 0))

        assert batch.x.shape[1] == 81

    def test_rollouts_are_deterministic(self):
        a = collect(Rollout(3001, SMALL, None, 1.0, "sample", 5))
        b = collect(Rollout(3001, SMALL, None, 1.0, "sample", 5))

        assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)

    @pytest.mark.parametrize("arch", ["mlp", "cnn"])
    def test_fit_returns_a_net_that_predicts_valid_actions(self, arch):
        cfg = ProbeConfig(grid_size=9, arch=arch, epochs=1, max_steps=60, hidden=(8,))
        batch = collect(Rollout(3000, cfg, None, 1.0, "sample", 0))

        net, _ = fit(None, batch, cfg, 0)

        assert set(net.predict(batch.x)) <= {0, 1, 2, 3, 4}


@pytest.mark.unit
class TestProbeRun:
    def test_probe_policy_plays_through_evaluate_deterministically(self):
        kwargs = dict(
            workers=1,
            fit_seeds=[3000, 3001],
            holdout_seeds=[0],
            eval_fit_seeds=[3000],
            eval_validation_seeds=[1000],
            log=False,
        )
        _, policy = run_probe(SMALL, **kwargs)

        first = evaluate(policy, [1000], max_steps=60, frame_skip=4)
        second = evaluate(policy, [1000], max_steps=60, frame_skip=4)

        assert first == second

    def test_policy_reads_the_window_it_was_trained_on(self):
        _, policy = run_probe(
            SMALL,
            workers=1,
            fit_seeds=[3000],
            holdout_seeds=[0],
            eval_fit_seeds=[3000],
            eval_validation_seeds=[1000],
            log=False,
        )

        assert (policy.grid_size, policy.cell_px) == (SMALL.grid_size, SMALL.cell_px)


@pytest.mark.unit
class TestGateReading:
    @pytest.mark.parametrize(
        "scores, word",
        [
            ({"a": 700.0, "b": 100.0}, "pass"),
            ({"a": 200.0}, "fail"),
            ({"a": 400.0}, "inconclusive"),
        ],
    )
    def test_g1_reading(self, scores, word):
        assert g1_reading(scores).startswith(word)
