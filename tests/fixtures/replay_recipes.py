"""Recipes for the committed regression replays in ``tests/fixtures/replays/``."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from infinite_maze.core.replay import ReplayRecorder
from infinite_maze.core.simulation import Intent, Simulation
from tests.fixtures.golden_harness import Script, fuzzScript

REPLAY_DIR = Path(__file__).parent / "replays"

KEY_INTENTS = {"R": (1, 0), "L": (-1, 0), "U": (0, -1), "D": (0, 1)}

# name -> (seed, start_pace, key script)
RECIPES: Dict[str, Tuple[int, int, Script]] = {
    "walk_right_blocked": (5, 0, [("R", 300)]),
    "fuzz_pace2": (1, 2, fuzzScript(1, 600)),
    "idle_death_pace1": (2, 1, [("", 30)]),
}


def scriptToIntents(script: Script) -> List[Intent]:
    """Convert a key script into one ``(dx, dy)`` intent per frame."""
    intents: List[Intent] = []
    for keys, count in script:
        dx = sum(KEY_INTENTS[k][0] for k in keys)
        dy = sum(KEY_INTENTS[k][1] for k in keys)
        intents.extend([(dx, dy)] * count)
    return intents


def buildReplay(seed: int, startPace: int, script: Script) -> Dict[str, Any]:
    """Play ``script`` like the human loop would, stopping when the player dies."""
    sim = Simulation(seed=seed, start_pace=startPace, max_steps=None)
    recorder = ReplayRecorder(seed, startPace)
    for intent in scriptToIntents(script):
        result = sim.step(intent)
        recorder.record(intent)
        if result.terminated:
            break
    return recorder.toReplay(sim.getState())
