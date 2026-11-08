"""Command-line entry point: ``infinite-maze`` / ``python -m infinite_maze``."""

import argparse
import json
import sys
from typing import List, Optional

from .core.engine import maze
from .core.replay import ReplayError, compareExpected, loadReplay, runReplay
from .utils.config import config


def buildParser() -> argparse.ArgumentParser:
    """Create the argument parser with all supported options."""
    parser = argparse.ArgumentParser(
        prog="infinite-maze", description="Infinite Maze game"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for maze generation; the same seed repeats the same maze",
    )
    parser.add_argument(
        "--start-pace",
        type=int,
        default=config.START_PACE,
        help="initial pace (global speed multiplier), for debugging",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        default=None,
        help="write a replay file per episode (PATH, PATH-2, PATH-3, ...)",
    )
    parser.add_argument(
        "--replay",
        metavar="PATH",
        default=None,
        help="play back a recorded replay instead of playing",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="with --replay: run without a window, print the final state and "
        "exit non-zero if it differs from the replay's expected state",
    )
    return parser


def parseArgs(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse and validate arguments; invalid combinations exit with status 2."""
    parser = buildParser()
    args = parser.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed < config.SEED_RANGE:
        parser.error(f"--seed must be between 0 and {config.SEED_RANGE - 1}")
    if args.start_pace < 0:
        parser.error("--start-pace must not be negative")
    if args.headless and args.replay is None:
        parser.error("--headless is only valid together with --replay")
    if args.replay is not None and (
        args.seed is not None
        or args.record is not None
        or args.start_pace != config.START_PACE
    ):
        parser.error(
            "--replay cannot be combined with --seed, --start-pace or --record"
        )
    return args


def playReplay(path: str, headless: bool) -> int:
    """Play a replay file; return 0 on success, 1 on mismatch, 2 on a bad file."""
    try:
        replay = loadReplay(path)
        result = runReplay(replay, render=not headless)
    except ReplayError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    mismatches = compareExpected(replay, result)
    for mismatch in mismatches:
        print(f"mismatch - {mismatch}", file=sys.stderr)
    return 1 if mismatches else 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run the game or a replay and return the process exit code."""
    args = parseArgs(argv)
    if args.replay is not None:
        return playReplay(args.replay, args.headless)
    maze(seed=args.seed, start_pace=args.start_pace, record_path=args.record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
