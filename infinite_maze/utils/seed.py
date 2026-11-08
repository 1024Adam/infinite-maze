"""Seed helpers shared by the maze generator, simulation and replays."""

import random

from .config import config


def newSeed() -> int:
    """Return a fresh, OS-entropy backed seed."""
    return random.SystemRandom().randrange(config.SEED_RANGE)
