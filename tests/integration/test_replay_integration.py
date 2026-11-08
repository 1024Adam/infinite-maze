"""Recording from the human loop, and CLI behaviour around replays."""

import json
from unittest.mock import patch

import pygame
import pytest

from infinite_maze import __main__ as cli
from infinite_maze.core.game import Game
from infinite_maze.core.replay import loadReplay, runReplay, saveReplay
from tests.fixtures.golden_harness import KeyState, runHuman
from tests.fixtures.pygame_mocks import PygameMockContext
from tests.fixtures.replay_recipes import RECIPES, REPLAY_DIR


class TestRecordFromHumanLoop:
    """``maze(record_path=...)`` writes replays that reproduce the session."""
    def test_recorded_session_replays_to_the_same_final_state(self, tmp_path):
        path = str(tmp_path / "run.json")

        trace = runHuman(
            5, [("R", 40), ("RD", 15), ("L", 2), ("", 5)], record_path=path
        )

        replay = loadReplay(path)
        end = trace[-1]
        assert end["label"] == "end"
        result = runReplay(replay)
        assert (result["score"], result["player"]) == (
            end["score"],
            [end["x"], end["y"]],
        )
        assert replay["expected"] == result

    def test_each_episode_is_written_to_its_own_file(self, tmp_path):
        path = str(tmp_path / "run.json")

        runHuman(
            1,
            [("R", 3), ("L", 5)],
            endResponses=[[pygame.K_y]],
            tailFrames=[("R", 20)],
            record_path=path,
        )

        first = loadReplay(path)
        second = loadReplay(str(tmp_path / "run-2.json"))
        assert first["expected"]["terminated"] is True
        assert second["seed"] == first["seed"]
        assert second["expected"]["terminated"] is False
        assert second["intents"] == [[1, 0, 20], [0, 0, 1]]  # 20 steps + ESC frame
        assert runReplay(second) == second["expected"]

    def test_nothing_is_written_without_record_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        runHuman(5, [("R", 5)])

        assert list(tmp_path.iterdir()) == []


class TestRenderedPlayback:
    """``runReplay(render=True)`` with display, keys and sleeping mocked."""

    def play(self, replay, keys):
        """Run rendered playback with ``keys`` held; return (state, mocks)."""
        with PygameMockContext(mock_event=False) as mocks, patch(
            "pygame.key.get_pressed", return_value=keys
        ), patch("pygame.event.get", return_value=[]), patch.object(
            Game, "cleanup"
        ) as cleanup:
            result = runReplay(replay, render=True)
        cleanup.assert_called_once()
        return result, mocks

    def test_rendered_playback_matches_headless_and_throttles_fps(self):
        replay = loadReplay(str(REPLAY_DIR / "walk_right_blocked.json"))

        result, mocks = self.play(replay, KeyState())

        assert result == replay["expected"]
        assert mocks["display"]["flip"].call_count > 0
        assert mocks["time"]["delay"].call_count > 0

    def test_escape_stops_playback_early(self):
        replay = loadReplay(str(REPLAY_DIR / "walk_right_blocked.json"))

        result, _ = self.play(replay, KeyState([pygame.K_ESCAPE]))

        assert result["ticks"] == 0


class TestReplayCli:
    """Replay-related command-line behaviour: exit codes and option validation."""

    def run(self, argv, capsys):
        """Call ``cli.main`` and return (exit code, stdout, stderr)."""
        code = cli.main(argv)
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_headless_replay_prints_state_and_succeeds(self, capsys):
        path = REPLAY_DIR / "walk_right_blocked.json"

        code, out, _ = self.run(["--replay", str(path), "--headless"], capsys)

        assert code == 0
        assert json.loads(out) == loadReplay(str(path))["expected"]

    def test_tampered_expected_exits_non_zero(self, tmp_path, capsys):
        replay = loadReplay(str(REPLAY_DIR / "walk_right_blocked.json"))
        replay["expected"]["ticks"] += 1
        path = str(tmp_path / "bad.json")
        saveReplay(replay, path)

        code, _, err = self.run(["--replay", path, "--headless"], capsys)

        assert code == 1
        assert "ticks" in err

    def test_incompatible_replay_exits_with_error(self, tmp_path, capsys):
        replay = loadReplay(str(REPLAY_DIR / "walk_right_blocked.json"))
        replay["version"] = 99
        path = str(tmp_path / "old.json")
        saveReplay(replay, path)

        code, _, err = self.run(["--replay", path, "--headless"], capsys)

        assert code == 2
        assert "version" in err

    @pytest.mark.parametrize(
        "argv",
        [
            ["--headless"],
            ["--replay", "x.json", "--seed", "1"],
            ["--replay", "x.json", "--record", "y.json"],
            ["--replay", "x.json", "--start-pace", "2"],
            ["--start-pace", "-1"],
        ],
    )
    def test_invalid_combinations_are_rejected(self, argv):
        with pytest.raises(SystemExit) as excinfo:
            cli.parseArgs(argv)
        assert excinfo.value.code == 2

    def test_recipes_are_all_committed(self):
        assert {p.stem for p in REPLAY_DIR.glob("*.json")} == set(RECIPES)
