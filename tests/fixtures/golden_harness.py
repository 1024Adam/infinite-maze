"""
Characterization harness: drives the real ``engine.maze()`` / ``controlled_run()``
loops under mocks and records a per-frame trace of the game state.

The traces are compared against golden files so that refactors of the game loop
cannot silently change gameplay. The harness only relies on public behaviour
(``Game.updateScreen`` being called once per frame, ``pygame.key.get_pressed``
driving input) so it works both before and after the loop is moved into
``Simulation``.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

import pygame

from infinite_maze.core import engine
from infinite_maze.core.game import Game
from tests.fixtures.pygame_mocks import PygameMockContext

GOLDEN_DIR = Path(__file__).parent / "golden"

KEY_CODES = {
    "R": pygame.K_RIGHT,
    "L": pygame.K_LEFT,
    "U": pygame.K_UP,
    "D": pygame.K_DOWN,
}

Script = List[Tuple[str, int]]  # (keys such as "RD", repeat count)


class ScriptDone(Exception):
    """Raised by the scripted wrapper once its actions are exhausted."""


class KeyState:
    """Stand-in for ``pygame.key.get_pressed()``: indexable by key constant."""

    def __init__(self, pressed: Sequence[int] = ()) -> None:
        self._pressed = frozenset(pressed)

    def __getitem__(self, key: int) -> bool:
        return key in self._pressed


def expandScript(script: Script) -> List[KeyState]:
    """Expand ``(keys, count)`` script entries into one KeyState per frame."""
    frames: List[KeyState] = []
    for keys, count in script:
        frames.extend(KeyState([KEY_CODES[k] for k in keys]) for _ in range(count))
    return frames


def fuzzScript(seed: int, frames: int, holdMax: int = 20) -> Script:
    """Deterministic pseudo-random key script biased towards moving right."""
    rng = random.Random(seed)
    options = ["R", "R", "R", "R", "RU", "RD", "U", "D", "L", "LU", "LD", ""]
    script: Script = []
    remaining = frames
    while remaining > 0:
        keys = rng.choice(options)
        limit = 2 if "L" in keys else holdMax
        count = min(rng.randint(1, limit), remaining)
        script.append((keys, count))
        remaining -= count
    return script


def linesHash(lines: Sequence[Any]) -> str:
    """Short hash of the sorted wall coordinates, to compare mazes cheaply."""
    coords = sorted(
        (line.getXStart(), line.getYStart(), line.getXEnd(), line.getYEnd())
        for line in lines
    )
    return hashlib.sha1(json.dumps(coords).encode()).hexdigest()[:10]


class _Recorder:
    """Collects the per-frame trace and applies the scenario's fixed pace."""

    def __init__(self, pace: int) -> None:
        self.pace = pace
        self.game: Optional[Game] = None
        self.player: Any = None
        self.lines: Any = None
        self.trace: List[Dict[str, Any]] = []
        self.frame = 0

    def snapshot(self, label: str) -> Dict[str, Any]:
        """Capture player position, score, pace and maze hash under ``label``."""
        assert self.game is not None and self.player is not None
        return {
            "label": label,
            "x": self.player.getX(),
            "y": self.player.getY(),
            "score": self.game.getScore(),
            "pace": self.game.getPace(),
            "lines": linesHash(self.lines),
        }

    def updateScreen(self, game: Game, player: Any, lines: Any) -> None:
        """Replacement for ``Game.updateScreen``: tick the clock and record a frame."""
        self.game, self.player, self.lines = game, player, lines
        game.getClock().update()
        if game.getClock().getTicks() == 1:
            # First frame of an episode: apply the scenario's fixed pace.
            game.setPace(self.pace)
        entry = self.snapshot("frame")
        entry["frame"] = self.frame
        self.frame += 1
        self.trace.append(entry)


def runHuman(
    seed: int,
    script: Script,
    pace: int = 0,
    endResponses: Sequence[Sequence[int]] = (),
    tailFrames: Optional[Script] = None,
    **mazeKwargs: Any,
) -> List[Dict[str, Any]]:
    """Drive ``engine.maze(seed=...)`` with scripted keys.

    ``script`` frames are followed by an ESC frame. When a game ends, the next
    entry of ``endResponses`` is returned from ``get_pressed`` (``[K_y]``
    restarts); once exhausted ``[K_n]`` quits. ``tailFrames`` play after a restart.
    Extra keyword arguments (for example ``record_path``) go to ``maze()``.
    """
    recorder = _Recorder(pace)
    queue = expandScript(script)
    pending = list(endResponses)
    tail = expandScript(tailFrames or [])
    state = {"ended": False, "calls": 0, "restarted": False}
    esc = KeyState([pygame.K_ESCAPE])

    def getPressed() -> KeyState:
        state["calls"] += 1
        if state["calls"] == 1:
            return KeyState()
        if state["ended"]:
            state["ended"] = False
            if pending:
                response = pending.pop(0)
                if pygame.K_y in response:
                    queue[:] = tail
                    state["restarted"] = True
                return KeyState(response)
            return KeyState([pygame.K_n])
        if queue:
            return queue.pop(0)
        return esc

    def printEndDisplay(game: Game) -> None:
        state["ended"] = True
        recorder.trace.append(recorder.snapshot("end"))

    with PygameMockContext(mock_event=False) as _mocks, patch.object(
        Game, "updateScreen", autospec=True, side_effect=recorder.updateScreen
    ), patch.object(
        Game, "printEndDisplay", autospec=True, side_effect=printEndDisplay
    ), patch(
        "pygame.key.get_pressed", side_effect=getPressed
    ), patch(
        "pygame.event.get", return_value=[]
    ):
        try:
            engine.maze(seed=seed, **mazeKwargs)
        except SystemExit:
            pass
    return recorder.trace


class ScriptedWrapper:
    """Controller for ``controlled_run`` that replays fixed actions and traces."""

    def __init__(self, recorder: _Recorder, actions: Sequence[int]) -> None:
        self.recorder = recorder
        self.actions = list(actions)
        self.gameoverScore: Optional[int] = None

    def control(self, values: Dict[str, Any]) -> int:
        """Record the frame's values and return the next action (or stop)."""
        entry = self.recorder.snapshot("control")
        entry.update(values)
        self.recorder.trace.append(entry)
        if not self.actions:
            raise ScriptDone()
        return self.actions.pop(0)

    def gameover(self, score: int) -> None:
        """Record the final score reported by ``controlled_run``."""
        self.gameoverScore = score
        entry = self.recorder.snapshot("gameover")
        entry["gameover_score"] = score
        self.recorder.trace.append(entry)


def fuzzActions(seed: int, count: int, holdMax: int = 20) -> List[int]:
    """Deterministic pseudo-random discrete actions biased towards RIGHT."""
    rng = random.Random(seed)
    options = [1, 1, 1, 1, 1, 3, 4, 2, 0]
    actions: List[int] = []
    while len(actions) < count:
        action = rng.choice(options)
        limit = 2 if action == 2 else holdMax
        actions.extend([action] * min(rng.randint(1, limit), count - len(actions)))
    return actions


def runControlled(
    seed: int,
    actions: Sequence[int],
    pace: int = 0,
    heldKeys: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Drive ``engine.controlled_run`` with scripted wrapper actions.

    ``heldKeys`` are pressed every frame and override the wrapper's action.
    """
    recorder = _Recorder(pace)
    wrapper = ScriptedWrapper(recorder, actions)
    with PygameMockContext(mock_event=False) as _mocks, patch.object(
        Game, "updateScreen", autospec=True, side_effect=recorder.updateScreen
    ), patch(
        "pygame.key.get_pressed",
        return_value=KeyState([KEY_CODES[k] for k in heldKeys]),
    ), patch("pygame.event.get", return_value=[]):
        try:
            engine.controlled_run(wrapper, 0, seed=seed)
        except ScriptDone:
            pass
    return recorder.trace


def goldenPath(name: str) -> Path:
    """Path of the golden trace file for scenario ``name``."""
    return GOLDEN_DIR / f"{name}.json"


def loadGolden(name: str) -> List[Dict[str, Any]]:
    """Load the committed trace for scenario ``name``."""
    with open(goldenPath(name)) as handle:
        trace: List[Dict[str, Any]] = json.load(handle)["trace"]
    return trace


def saveGolden(name: str, description: str, trace: List[Dict[str, Any]]) -> None:
    """Write a trace, one entry per line so diffs stay readable."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    rows = ",\n".join(json.dumps(entry, separators=(",", ":")) for entry in trace)
    with open(goldenPath(name), "w") as handle:
        handle.write(
            '{"description":' + json.dumps(description) + ',"trace":[\n'
            + rows
            + "\n]}\n"
        )
