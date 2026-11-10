"""Tests for ``training.train`` (skipped without the ``ml-train`` group)."""

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

from training.evaluate import evaluate  # noqa: E402
from training.experiment import ExperimentConfig, load_experiment_config  # noqa: E402
from training.model_policy import ModelPolicy  # noqa: E402
from training.train import METRIC_COLUMNS, prepare_run_dir, train  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[2] / "training" / "configs"
SMOKE = CONFIG_DIR / "smoke.json"
SMOKE_V2 = CONFIG_DIR / "smoke_v2.json"
SMOKE_V3 = CONFIG_DIR / "smoke_v3.json"


def tiny_config(**overrides):
    values = {"name": "t", "train_seeds": [0], "total_timesteps": 10}
    values.update(overrides)
    return ExperimentConfig.from_dict(values)


@pytest.mark.unit
class TestRunDirectory:
    def test_new_run_gets_the_expected_layout_and_a_config_copy(self, tmp_path):
        cfg = tiny_config()
        run = tmp_path / "t"

        prepare_run_dir(run, cfg, resume=False)

        assert (run / "checkpoints").is_dir() and (run / "replays").is_dir()
        assert load_experiment_config(run / "config.json") == cfg

    def test_existing_run_is_never_overwritten(self, tmp_path):
        run = tmp_path / "t"
        prepare_run_dir(run, tiny_config(), resume=False)

        with pytest.raises(FileExistsError):
            prepare_run_dir(run, tiny_config(), resume=False)

    def test_resume_needs_a_config(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            prepare_run_dir(tmp_path / "nope", tiny_config(), resume=True)


@pytest.mark.slow
@pytest.mark.integration
class TestSmokeRun:
    @pytest.mark.parametrize(
        "path", [SMOKE, SMOKE_V2, SMOKE_V3], ids=["v1", "v2_shaped", "v3_window31"]
    )
    def test_smoke_run_writes_every_artefact_and_the_model_is_evaluable(
        self, tmp_path, path
    ):
        cfg = load_experiment_config(path)
        run = tmp_path / cfg.name

        train(cfg, run)

        for name in (
            "config.json",
            "metadata.json",
            "metrics.csv",
            "best.zip",
            "final.zip",
            "eval.json",
        ):
            assert (run / name).is_file(), name
        assert list((run / "checkpoints").glob("*.zip"))
        assert list((run / "replays" / "final").glob("*.json"))
        metadata = json.loads((run / "metadata.json").read_text())
        assert {"git_commit", "libraries", "python", "platform", "seed"} <= set(
            metadata
        )
        with open(run / "metrics.csv") as handle:
            rows = list(csv.reader(handle))
        assert tuple(rows[0]) == METRIC_COLUMNS and len(rows) > 1
        history = json.loads((run / "eval.json").read_text())["history"]
        assert set(history[-1]["sets"]) == set(cfg.eval_seeds)

        report = evaluate(
            ModelPolicy.load(
                str(run / "final.zip"),
                cfg.observation.grid_size,
                cfg.observation.cell_px,
            ),
            [0],
            max_steps=cfg.max_steps,
            frame_skip=cfg.frame_skip,
        )
        assert report.episodes[0].steps > 0


@pytest.mark.unit
class TestTrainingEnvShaping:
    def test_training_env_is_shaped_and_scaled_from_the_config(self):
        from training.train import make_training_env

        cfg = load_experiment_config(SMOKE_V2)
        env = make_training_env(cfg, cfg.start_pace)
        try:
            env.reset()
            _, rewards, _, infos = env.step([1] * cfg.n_envs)
        finally:
            env.close()

        assert cfg.shaping.weight > 0 and cfg.train_reward_scale == 0.05
        for reward, info in zip(rewards, infos):
            expected = (info["base_reward"] + info["shaping_reward"]) * 0.05
            assert reward == pytest.approx(expected, rel=1e-5)
