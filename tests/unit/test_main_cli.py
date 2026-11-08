"""Argument parsing for the ``infinite-maze`` entry point (no pygame window)."""

from unittest.mock import patch

import pytest

from infinite_maze import __main__ as cli
from infinite_maze.utils.config import config


class TestParseArgs:
    """Option parsing and validation."""
    def test_seed_defaults_to_none(self):
        assert cli.parseArgs([]).seed is None

    def test_seed_is_parsed(self):
        assert cli.parseArgs(["--seed", "1234"]).seed == 1234

    @pytest.mark.parametrize("bad", ["-1", str(config.SEED_RANGE), "abc"])
    def test_invalid_seed_is_rejected(self, bad):
        with pytest.raises(SystemExit) as excinfo:
            cli.parseArgs(["--seed", bad])
        assert excinfo.value.code == 2


class TestMain:
    """``main`` forwards parsed options to ``maze``."""
    def test_main_passes_seed_to_maze(self):
        with patch.object(cli, "maze") as mock_maze:
            assert cli.main(["--seed", "7"]) == 0
        mock_maze.assert_called_once_with(
            seed=7, start_pace=config.START_PACE, record_path=None
        )

    def test_main_passes_start_pace_and_record_path(self):
        with patch.object(cli, "maze") as mock_maze:
            cli.main(["--start-pace", "3", "--record", "replays/run.json"])
        mock_maze.assert_called_once_with(
            seed=None, start_pace=3, record_path="replays/run.json"
        )
