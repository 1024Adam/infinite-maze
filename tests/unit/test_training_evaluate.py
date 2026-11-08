"""Unit tests for ``training.evaluate`` that do not need a real maze."""

import json
from unittest.mock import patch

import pytest

from training import config
from training import evaluate as ev
from training.policies import AlwaysRightPolicy
from training.reward import RewardConfig


def episode(seed, total_return, terminated=False, score=10, steps=100):
    """An EpisodeResult with convenient defaults."""
    return ev.EpisodeResult(
        seed=seed,
        score=score,
        steps=steps,
        total_return=total_return,
        terminated=terminated,
        truncated=not terminated,
    )


def report(name="p", episodes=None):
    """A report built from hand-made episodes."""
    episodes = episodes or [
        episode(1, 10.0),
        episode(2, -30.0, terminated=True, score=2, steps=40),
        episode(3, 20.0, score=30, steps=160),
    ]
    return ev.EvalReport(name, 200, 0, RewardConfig(), episodes)


class TestReport:
    def test_summary_statistics(self):
        s = report().summary()

        assert s["mean_return"] == pytest.approx(0.0)
        assert s["median_return"] == 10.0
        assert (s["min_return"], s["max_return"]) == (-30.0, 20.0)
        assert s["mean_score"] == pytest.approx(14.0)
        assert s["mean_steps"] == pytest.approx(100.0)
        assert s["death_rate"] == pytest.approx(1 / 3)

    def test_worst_seed_is_the_lowest_return(self):
        assert report().worst_seed() == 2

    def test_to_dict_is_json_serialisable_and_names_the_return(self):
        data = report().to_dict()

        json.dumps(data)
        assert data["episodes"][0]["return"] == 10.0
        assert "total_return" not in data["episodes"][0]
        assert data["reward"]["death_penalty"] == config.REWARD_DEATH_PENALTY

    def test_table_lists_every_policy(self):
        table = ev.format_table([report("alpha"), report("beta")])

        assert "alpha" in table and "beta" in table
        assert "death rate" in table


class TestEvaluateValidation:
    def test_empty_seeds_rejected(self):
        with pytest.raises(ValueError, match="seeds"):
            ev.evaluate(AlwaysRightPolicy(), [])

    def test_non_positive_max_steps_rejected(self):
        with pytest.raises(ValueError, match="max_steps"):
            ev.evaluate(AlwaysRightPolicy(), [1], max_steps=0)

    def test_reward_config_that_makes_dying_attractive_is_rejected(self):
        bad = RewardConfig(death_penalty=1.0)

        with pytest.raises(ValueError, match="death_penalty"):
            ev.evaluate(AlwaysRightPolicy(), [1], max_steps=1000, reward=bad)


class TestCli:
    def test_seed_sets_are_the_agreed_ranges(self):
        assert list(config.TRAIN_SEEDS) == list(range(0, 100))
        assert list(config.VALIDATION_SEEDS) == list(range(1000, 1050))
        assert list(config.TEST_SEEDS) == list(range(2000, 2050))

    def test_parser_defaults(self):
        args = ev.build_parser().parse_args([])

        assert args.policy == "all"
        assert args.seeds == "validation"
        assert args.max_steps == config.EVAL_MAX_STEPS

    def test_main_runs_all_baselines_prints_a_table_and_writes_json(
        self, tmp_path, capsys
    ):
        out = tmp_path / "runs" / "baselines.json"
        calls = []

        def fake_evaluate(policy, seeds, max_steps, start_pace, reward, frame_skip):
            calls.append((policy.name, list(seeds), max_steps, start_pace))
            return report(policy.name)

        with patch.object(ev, "evaluate", side_effect=fake_evaluate):
            code = ev.main(
                ["--seeds", "validation", "--num-seeds", "3", "--out", str(out)]
            )

        assert code == 0
        assert [c[0] for c in calls] == ["random", "right", "heuristic"]
        assert calls[0][1] == [1000, 1001, 1002]
        printed = capsys.readouterr().out
        assert "heuristic" in printed
        saved = json.loads(out.read_text())
        assert [r["policy"] for r in saved["reports"]] == [
            "random",
            "right",
            "heuristic",
        ]

    def test_main_writes_the_worst_seed_replay(self, tmp_path, capsys):
        replay = {"version": 1}
        with patch.object(ev, "evaluate", return_value=report("right")), patch.object(
            ev, "record_replay", return_value=replay
        ) as record, patch.object(ev, "saveReplay") as save:
            ev.main(
                [
                    "--policy",
                    "right",
                    "--num-seeds",
                    "2",
                    "--worst-replay-dir",
                    str(tmp_path),
                ]
            )

        assert record.call_args[0][1] == 2  # the worst seed
        save.assert_called_once()
        assert save.call_args[0][1].endswith("right-worst-seed2.json")
