"""
Record and replay of games.

A replay is a small, versioned JSON document: ``(seed, start_pace, intents)``
fully determines a game, because the rules are tick based and the maze is
seeded. ``expected`` optionally stores the final state, which turns a replay
into a regression test.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pygame

from ..entities.player import Player
from ..utils.config import config
from .game import Game
from .simulation import Intent, Simulation

REPLAY_VERSION = 1

RunLength = List[int]  # [dx, dy, count]


class ReplayError(Exception):
    """Raised when a replay file is malformed or incompatible."""


def mazeConfig() -> Dict[str, int]:
    """Maze dimensions recorded in a replay, used to detect incompatible builds."""
    return {
        "rows": config.MAZE_ROWS,
        "cols": config.MAZE_COLS,
        "cell": config.MAZE_CELL_SIZE,
    }


def decodeIntents(runs: List[RunLength]) -> List[Intent]:
    """Expand run-length encoded ``[dx, dy, count]`` runs into one intent per step."""
    intents: List[Intent] = []
    for run in runs:
        if len(run) != 3 or run[2] < 0:
            raise ReplayError(f"invalid intent run: {run!r}")
        intents.extend([(run[0], run[1])] * run[2])
    return intents


class ReplayRecorder:
    """Accumulates the intents that reach ``Simulation.step()``."""

    def __init__(self, seed: int, startPace: int) -> None:
        """Start an empty recording for an episode with this seed and start pace."""
        self.seed = seed
        self.startPace = startPace
        self.runs: List[RunLength] = []

    def record(self, intent: Intent) -> None:
        """Append one step, extending the last run when the intent repeats."""
        dx, dy = intent
        if self.runs and self.runs[-1][0] == dx and self.runs[-1][1] == dy:
            self.runs[-1][2] += 1
        else:
            self.runs.append([dx, dy, 1])

    def toReplay(self, finalState: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build the replay document, with ``finalState`` as ``expected`` if given."""
        replay: Dict[str, Any] = {
            "version": REPLAY_VERSION,
            "seed": self.seed,
            "start_pace": self.startPace,
            "maze": mazeConfig(),
            "intents": [list(run) for run in self.runs],
        }
        if finalState is not None:
            replay["expected"] = finalState
        return replay


def episodePath(basePath: str, episode: int) -> str:
    """``replays/run.json`` for episode 1, ``replays/run-2.json`` for episode 2."""
    if episode <= 1:
        return basePath
    root, extension = os.path.splitext(basePath)
    return f"{root}-{episode}{extension}"


def saveReplay(replay: Dict[str, Any], path: str) -> None:
    """Write ``replay`` as JSON, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(replay, handle)
        handle.write("\n")


def loadReplay(path: str) -> Dict[str, Any]:
    """Read and validate a replay file; raises ``ReplayError`` if unusable."""
    try:
        with open(path) as handle:
            replay = json.load(handle)
    except (OSError, ValueError) as error:
        raise ReplayError(f"cannot read replay {path}: {error}") from error
    validateReplay(replay)
    return replay  # type: ignore[no-any-return]


def validateReplay(replay: Any) -> None:
    """Check structure, version and maze config; raises ``ReplayError`` on failure."""
    if not isinstance(replay, dict):
        raise ReplayError("replay must be a JSON object")
    version = replay.get("version")
    if version != REPLAY_VERSION:
        raise ReplayError(
            f"unsupported replay version {version!r} (expected {REPLAY_VERSION})"
        )
    for key in ("seed", "start_pace", "maze", "intents"):
        if key not in replay:
            raise ReplayError(f"replay is missing '{key}'")
    if replay["maze"] != mazeConfig():
        raise ReplayError(
            f"replay was recorded with maze {replay['maze']} but this build uses "
            f"{mazeConfig()}; it would diverge"
        )


def compareExpected(replay: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
    """Return human-readable mismatches between ``expected`` and ``result``."""
    expected = replay.get("expected")
    if expected is None:
        return []
    return [
        f"{key}: expected {value!r}, got {result.get(key)!r}"
        for key, value in expected.items()
        if result.get(key) != value
    ]


def runReplay(replay: Dict[str, Any], render: bool = False) -> Dict[str, Any]:
    """Play ``replay`` and return the final simulation state.

    With ``render`` the game is shown in a window at normal speed; closing the
    window or pressing ESC stops early. Without it no display is needed.
    """
    validateReplay(replay)
    intents = decodeIntents(replay["intents"])

    game = Game(headless=not render)
    player = Player(config.PLAYER_START_X, config.PLAYER_START_Y, headless=not render)
    sim = Simulation(
        seed=replay["seed"],
        start_pace=replay["start_pace"],
        max_steps=None,
        game=game,
        player=player,
    )

    if not render:
        for intent in intents:
            sim.step(intent)
        return sim.getState()

    for intent in intents:
        if _shouldStop():
            break
        game.updateScreen(player, sim.getLines())
        sim.step(intent)
        game.limitFps()
    game.updateScreen(player, sim.getLines())
    state = sim.getState()
    game.cleanup()
    return state


def _shouldStop() -> bool:
    """Whether the user pressed ESC or closed the window during rendered playback."""
    if pygame.key.get_pressed()[pygame.K_ESCAPE]:
        return True
    return any(event.type == pygame.QUIT for event in pygame.event.get())
