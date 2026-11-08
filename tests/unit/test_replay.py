"""Unit tests for replay recording, loading and playback."""

import json
import os
from unittest.mock import patch

import pytest

from infinite_maze.core import replay as replay_module
from infinite_maze.core.replay import (
    REPLAY_VERSION,
    ReplayError,
    ReplayRecorder,
    compareExpected,
    decodeIntents,
    episodePath,
    loadReplay,
    runReplay,
    saveReplay,
)
from tests.fixtures.replay_recipes import RECIPES, REPLAY_DIR, buildReplay


def recorded(intents, seed=1, startPace=0):
    """Build a replay document by recording ``intents``."""
    recorder = ReplayRecorder(seed, startPace)
    for intent in intents:
        recorder.record(intent)
    return recorder.toReplay()


class TestRunLengthEncoding:
    """Intents are stored as run-length encoded ``[dx, dy, count]`` runs."""
    def test_empty(self):
        assert recorded([])["intents"] == []

    def test_single_step(self):
        assert recorded([(1, 0)])["intents"] == [[1, 0, 1]]

    def test_long_run_is_one_entry(self):
        assert recorded([(1, 0)] * 10000)["intents"] == [[1, 0, 10000]]

    def test_changes_start_new_runs(self):
        intents = [(1, 0), (1, 0), (0, 1), (1, 0), (0, 0), (0, 0)]

        runs = recorded(intents)["intents"]

        assert runs == [[1, 0, 2], [0, 1, 1], [1, 0, 1], [0, 0, 2]]
        assert decodeIntents(runs) == intents

    @pytest.mark.parametrize("bad", [[1, 0], [1, 0, -1], [1, 0, 1, 1]])
    def test_malformed_run_is_rejected(self, bad):
        with pytest.raises(ReplayError):
            decodeIntents([bad])


class TestFiles:
    """Saving, loading and validating replay files."""
    def test_episode_paths_insert_counter_before_extension(self):
        assert episodePath("replays/run.json", 1) == "replays/run.json"
        assert episodePath("replays/run.json", 2) == "replays/run-2.json"
        assert episodePath("run", 3) == "run-3"

    def test_save_load_round_trip_creates_directories(self, tmp_path):
        path = str(tmp_path / "nested" / "run.json")
        replay = recorded([(1, 0)] * 3 + [(0, 1)], seed=9, startPace=2)

        saveReplay(replay, path)

        assert loadReplay(path) == replay
        assert replay["version"] == REPLAY_VERSION

    def test_unreadable_file_is_a_replay_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")

        with pytest.raises(ReplayError, match="cannot read"):
            loadReplay(str(path))
        with pytest.raises(ReplayError, match="cannot read"):
            loadReplay(str(tmp_path / "missing.json"))

    def test_wrong_version_is_rejected(self, tmp_path):
        replay = recorded([])
        replay["version"] = REPLAY_VERSION + 1
        path = str(tmp_path / "v.json")
        saveReplay(replay, path)

        with pytest.raises(ReplayError, match="version"):
            loadReplay(path)

    def test_maze_config_mismatch_is_rejected(self, tmp_path):
        replay = recorded([])
        replay["maze"]["cell"] += 1
        path = str(tmp_path / "m.json")
        saveReplay(replay, path)

        with pytest.raises(ReplayError, match="diverge"):
            loadReplay(path)

    def test_missing_field_is_rejected(self, tmp_path):
        replay = recorded([])
        del replay["seed"]
        path = str(tmp_path / "s.json")
        saveReplay(replay, path)

        with pytest.raises(ReplayError, match="seed"):
            loadReplay(path)


class TestPlayback:
    """Headless playback reproduces the recorded final state."""
    def test_round_trip_reproduces_expected_state(self, tmp_path):
        replay = buildReplay(5, 0, [("R", 40), ("RD", 10), ("L", 3)])
        path = str(tmp_path / "run.json")
        saveReplay(replay, path)

        result = runReplay(loadReplay(path))

        assert result == replay["expected"]
        assert compareExpected(replay, result) == []

    def test_replay_without_expected_has_no_mismatches(self):
        replay = recorded([(1, 0)] * 5, seed=3)

        result = runReplay(replay)

        assert "expected" not in replay
        assert result["ticks"] == 5
        assert compareExpected(replay, result) == []

    def test_tampered_expected_is_reported(self):
        replay = buildReplay(5, 0, [("R", 30)])
        replay["expected"]["score"] += 1

        mismatches = compareExpected(replay, runReplay(replay))

        assert len(mismatches) == 1
        assert mismatches[0].startswith("score")

    def test_playback_does_not_sleep_or_render_when_headless(self):
        replay = recorded([(1, 0)] * 5, seed=3)

        with patch("pygame.time.delay") as delay, patch(
            "pygame.display.set_mode"
        ) as set_mode:
            runReplay(replay, render=False)

        delay.assert_not_called()
        set_mode.assert_not_called()


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_committed_regression_replay(name):
    path = REPLAY_DIR / f"{name}.json"
    if os.environ.get("REGENERATE_GOLDENS"):
        seed, startPace, script = RECIPES[name]
        saveReplay(buildReplay(seed, startPace, script), str(path))

    replay = loadReplay(str(path))
    assert "expected" in replay
    assert compareExpected(replay, runReplay(replay)) == []


def test_committed_replays_are_valid_json_documents():
    for path in REPLAY_DIR.glob("*.json"):
        with open(path) as handle:
            assert json.load(handle)["version"] == replay_module.REPLAY_VERSION
