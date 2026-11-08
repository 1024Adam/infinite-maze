"""Scenario definitions for the golden-trace characterization tests."""

from typing import Any, Callable, Dict, List, Tuple

import pygame

from tests.fixtures.golden_harness import (
    fuzzActions,
    fuzzScript,
    runControlled,
    runHuman,
)

Trace = List[Dict[str, Any]]
Scenario = Tuple[str, Callable[[], Trace]]

# Seed 21 has a 14-cell straight corridor in maze row 1, reachable by walking
# up the free margin left of the maze (the y offset within a row is preserved).
CORRIDOR_SEED = 21
ROWS_TO_CORRIDOR = 7
ROW_HEIGHT_FRAMES = 22

SCENARIOS: Dict[str, Scenario] = {
    "walk_right_until_blocked": (
        "Hold RIGHT at pace 0 until a wall stops the player.",
        lambda: runHuman(5, [("R", 300)]),
    ),
    "slide_along_wall": (
        "Hold RIGHT+DOWN then RIGHT+UP: diagonal input slides along walls.",
        lambda: runHuman(3, [("RD", 120), ("RU", 120), ("R", 20)]),
    ),
    "clamp_at_x_max": (
        "Walk up the margin, then right down a long corridor past X_MAX.",
        lambda: runHuman(
            CORRIDOR_SEED,
            [("U", ROWS_TO_CORRIDOR * ROW_HEIGHT_FRAMES), ("R", 360)],
        ),
    ),
    "left_score_floor": (
        "Move right then left past the start: score floors at 0, then death.",
        lambda: runHuman(1, [("R", 3), ("L", 5)]),
    ),
    "idle_death_pace1": (
        "Do nothing at pace 1: scrolling pushes the player past X_MIN.",
        lambda: runHuman(2, [("", 30)], pace=1),
    ),
    "restart_after_death": (
        "Die, answer 'y', then play on; the restart respawns at PLAYER_START_Y.",
        lambda: runHuman(
            1,
            [("R", 3), ("L", 5)],
            endResponses=[[pygame.K_y]],
            tailFrames=[("R", 20), ("D", 10)],
        ),
    ),
    "fuzz_pace0": (
        "Pseudo-random human input at pace 0.",
        lambda: runHuman(8, fuzzScript(8, 600)),
    ),
    "fuzz_pace1_recycle": (
        "Pseudo-random human input at pace 1; lines get recycled.",
        lambda: runHuman(8, fuzzScript(18, 900), pace=1),
    ),
    "fuzz_pace2": (
        "Pseudo-random human input at pace 2.",
        lambda: runHuman(1, fuzzScript(1, 600), pace=2),
    ),
    "fuzz_pace3": (
        "Pseudo-random human input at pace 3.",
        lambda: runHuman(7, fuzzScript(7, 600), pace=3),
    ),
    "controlled_fuzz_pace0": (
        "controlled_run with scripted actions at pace 0.",
        lambda: runControlled(8, fuzzActions(8, 500)),
    ),
    "controlled_fuzz_pace2": (
        "controlled_run with scripted actions at pace 2.",
        lambda: runControlled(1, fuzzActions(1, 500), pace=2),
    ),
    "controlled_keyboard_override": (
        "A held RIGHT key overrides the wrapper's LEFT actions.",
        lambda: runControlled(5, [2] * 60, heldKeys=["R"]),
    ),
    "controlled_idle_death_pace1": (
        "controlled_run ends and reports gameover when scrolled past X_MIN.",
        lambda: runControlled(2, [0] * 100, pace=1),
    ),
}
