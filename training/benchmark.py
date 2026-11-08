"""Throughput benchmark for maze generation, reset and ``Simulation.step``.

    python -m training.benchmark [--seeds N] [--steps N] [--profile]

Prints a table (median and p95 per measurement) and, with ``--profile``, the
hot spots of a long scripted run ranked by cumulative time.
"""

import argparse
import cProfile
import io
import pstats
import random
import statistics
import sys
import time
from typing import Callable, List, Optional, Tuple

from infinite_maze.core.game import Game
from infinite_maze.core.simulation import Simulation
from infinite_maze.entities.maze import Line
from infinite_maze.utils.config import config as game_config

from . import config
from .observation import build_observation


def scripted_intents(seed: int) -> Callable[[], Tuple[int, int]]:
    """Cheap reproducible fuzz policy: mostly right, with vertical wandering."""
    rng = random.Random(seed)
    intents = [game_config.ACTION_TO_INTENT[a] for a in config.BENCH_ACTION_POOL]

    def next_intent() -> Tuple[int, int]:
        return intents[rng.randrange(len(intents))]

    return next_intent


def percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile of ``values`` (``fraction`` in [0, 1])."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def time_generate(seed: int) -> float:
    """Seconds for one ``Line.generateMaze`` call."""
    game = Game(headless=True)
    start = time.perf_counter()
    Line.generateMaze(
        game, game_config.MAZE_ROWS, game_config.MAZE_COLS, random.Random(seed)
    )
    return time.perf_counter() - start


def time_reset(sim: Simulation, seed: int) -> float:
    """Seconds for one ``Simulation.reset(seed)``."""
    start = time.perf_counter()
    sim.reset(seed)
    return time.perf_counter() - start


def steps_per_second(
    seed: int, steps: int, with_observation: bool, start_pace: int
) -> float:
    """Steps per second of a scripted run; maze resets after a death are excluded."""
    sim = Simulation(seed=seed, start_pace=start_pace, max_steps=None)
    next_intent = scripted_intents(seed)
    resetTime = 0.0
    start = time.perf_counter()
    for _ in range(steps):
        if with_observation:
            build_observation(sim)
        result = sim.step(next_intent())
        if result.terminated:
            resetTime += time_reset(sim, seed)
    elapsed = time.perf_counter() - start - resetTime
    return steps / elapsed


def profile_run(seed: int, steps: int, top: int) -> str:
    """``cProfile`` the scripted run (observation included), return the top rows."""
    profiler = cProfile.Profile()
    profiler.enable()
    steps_per_second(seed, steps, True, config.BENCH_START_PACE)
    profiler.disable()
    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer)
    stats.strip_dirs().sort_stats("cumulative").print_stats(top)
    return buffer.getvalue()


def summarise(label: str, values: List[float], unit: str, scale: float) -> str:
    """One table row: median and p95 of ``values``, scaled to ``unit``."""
    median = statistics.median(values) * scale
    p95 = percentile(values, 0.95) * scale
    return f"{label:<44}{median:>14.2f}{p95:>14.2f}  {unit}"


def run(seeds: int, steps: int, profile: bool) -> str:
    """Run every measurement and return the report text."""
    seed_list = list(range(seeds))
    lines = [f"{'measurement':<44}{'median':>14}{'p95':>14}"]
    lines.append("-" * 72)

    generate = [time_generate(s) for s in seed_list]
    lines.append(summarise("Line.generateMaze (new seeds)", generate, "ms", 1000))

    sim = Simulation(seed=seed_list[0], max_steps=None)
    sameSeed = [time_reset(sim, seed_list[0]) for _ in seed_list]
    lines.append(summarise("Simulation.reset (same seed)", sameSeed, "ms", 1000))
    newSeeds = [time_reset(sim, s) for s in seed_list]
    lines.append(summarise("Simulation.reset (new seeds)", newSeeds, "ms", 1000))

    for withObs in (False, True):
        rates = [
            steps_per_second(s, steps, withObs, config.BENCH_START_PACE)
            for s in seed_list
        ]
        label = "steps/s " + ("with observation" if withObs else "step only")
        lines.append(summarise(label, rates, "steps/s", 1))

    report = "\n".join(lines)
    if profile:
        report += f"\n\nProfile ({steps} steps, observation included):\n"
        report += profile_run(seed_list[0], steps, config.BENCH_PROFILE_ROWS)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m training.benchmark",
        description="Benchmark maze generation, reset and simulation steps.",
    )
    parser.add_argument("--seeds", type=int, default=config.BENCH_SEEDS)
    parser.add_argument("--steps", type=int, default=config.BENCH_STEPS)
    parser.add_argument("--profile", action="store_true", help="rank hot spots")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.seeds <= 0 or args.steps <= 0:
        raise SystemExit("--seeds and --steps must be positive")
    print(run(args.seeds, args.steps, args.profile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
