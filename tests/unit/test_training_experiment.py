"""Tests for ``training.experiment`` (stdlib only: no ML libraries needed)."""

import json
from pathlib import Path

import pytest

from training import config
from training.experiment import (
    ExperimentConfig,
    load_experiment_config,
    resolve_seeds,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "training" / "configs"


def valid(**overrides):
    values = {
        "name": "unit",
        "train_seeds": [0],
        "total_timesteps": 1000,
        "eval_seeds": {"train": [0], "validation": "validation"},
        "select_on": "train",
    }
    values.update(overrides)
    return values


@pytest.mark.unit
class TestParsing:
    def test_valid_config_round_trips(self):
        cfg = ExperimentConfig.from_dict(valid(reward={"step_penalty": 0.001}))

        again = ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))

        assert again == cfg
        assert cfg.reward.step_penalty == 0.001
        assert cfg.eval_seed_lists()["validation"] == list(config.VALIDATION_SEEDS)

    @pytest.mark.parametrize(
        "values",
        [
            valid(surprise=1),
            valid(reward={"step_penalty": 0.001, "bogus": 1}),
            valid(ppo={"learning_rate": 0.1, "bogus": 1}),
            valid(shaping={"weight": 1.0, "bogus": 1}),
            valid(curriculum=[{"timesteps": 0, "start_pace": 1, "bogus": 1}]),
        ],
    )
    def test_unknown_keys_are_rejected_at_every_level(self, values):
        with pytest.raises(ValueError, match="unknown"):
            ExperimentConfig.from_dict(values)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"total_timesteps": 0},
            {"frame_skip": 0},
            {"n_envs": 0},
            {"name": "a/b"},
            {"algo": "dqn"},
            {"select_on": "missing"},
            {"reward": {"death_penalty": 1.0, "step_penalty": 0.01}},
            {"curriculum": [{"timesteps": 5, "start_pace": 1}] * 2},
            {"shaping": {"weight": -1.0}},
            {"train_reward_scale": 0.0},
            {"shaping": {"weight": 44.0, "gamma": 0.9}},
        ],
    )
    def test_invalid_values_are_rejected(self, overrides):
        with pytest.raises(ValueError):
            ExperimentConfig.from_dict(valid(**overrides))

    def test_test_seeds_can_never_be_used(self):
        with pytest.raises(ValueError, match="test"):
            ExperimentConfig.from_dict(valid(train_seeds="test"))
        with pytest.raises(ValueError, match="test"):
            ExperimentConfig.from_dict(valid(eval_seeds={"x": [2001]}, select_on="x"))


@pytest.mark.unit
class TestSeedsAndCurriculum:
    def test_named_sets_resolve_to_the_fixed_benchmark(self):
        assert resolve_seeds("train") == list(range(0, 100))

    def test_unknown_set_name_is_rejected(self):
        with pytest.raises(ValueError):
            resolve_seeds("nope")

    def test_pace_follows_the_curriculum_stages(self):
        cfg = ExperimentConfig.from_dict(
            valid(
                start_pace=4,
                curriculum=[
                    {"timesteps": 0, "start_pace": 0},
                    {"timesteps": 500, "start_pace": 2},
                ],
            )
        )

        assert [cfg.pace_at(t) for t in (0, 499, 500, 900)] == [0, 0, 2, 2]

    def test_without_a_curriculum_pace_is_start_pace(self):
        assert ExperimentConfig.from_dict(valid(start_pace=3)).pace_at(10**6) == 3


@pytest.mark.unit
class TestShapingConfig:
    def test_shaping_round_trips(self):
        cfg = ExperimentConfig.from_dict(
            valid(shaping={"weight": 44.0, "gamma": 0.995}, train_reward_scale=0.05)
        )

        again = ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))

        assert again == cfg
        assert cfg.shaping.weight == 44.0 and cfg.train_reward_scale == 0.05

    def test_v1_config_loads_with_shaping_off(self):
        cfg = load_experiment_config(CONFIG_DIR / "v1_fixed_seed0.json")

        assert cfg.shaping.weight == 0.0
        assert cfg.train_reward_scale == 1.0

    def test_gamma_is_only_checked_when_shaping_is_on(self):
        ExperimentConfig.from_dict(
            valid(shaping={"weight": 0.0, "gamma": 0.9}, ppo={"gamma": 0.995})
        )

    def test_v2_configs_differ_only_by_shaping_weight(self):
        control = load_experiment_config(CONFIG_DIR / "v2a_multiseed_control.json")
        shaped = load_experiment_config(CONFIG_DIR / "v2b_multiseed_shaped.json")

        left, right = control.to_dict(), shaped.to_dict()
        for key in ("name", "shaping"):
            left.pop(key)
            right.pop(key)
        assert left == right
        assert (control.shaping.weight, shaped.shaping.weight) == (0.0, 44.0)
        assert control.select_on == shaped.select_on == "validation"


@pytest.mark.unit
class TestCommittedConfigs:
    @pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.json")))
    def test_every_committed_config_loads(self, path):
        cfg = load_experiment_config(path)

        assert cfg.name == path.stem


@pytest.mark.unit
class TestObservationConfig:
    def test_default_matches_the_constants_and_old_configs_load(self):
        cfg = load_experiment_config(CONFIG_DIR / "v1_fixed_seed0.json")

        assert (cfg.observation.grid_size, cfg.observation.cell_px) == (
            config.OBS_GRID_SIZE,
            config.OBS_GRID_CELL_PX,
        )

    def test_observation_round_trips(self):
        cfg = ExperimentConfig.from_dict(
            valid(observation={"grid_size": 31, "cell_px": 11})
        )

        again = ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))

        assert again == cfg and again.observation.grid_size == 31

    @pytest.mark.parametrize(
        "observation",
        [
            {"grid_size": 0},
            {"cell_px": 0},
            {"grid_size": 59, "cell_px": 11},  # 649 px: wider than the screen
            {"grid_size": 9, "bogus": 1},
        ],
    )
    def test_invalid_observation_is_rejected(self, observation):
        with pytest.raises(ValueError):
            ExperimentConfig.from_dict(valid(observation=observation))
