"""Oracle ceilings: what a perfect-information player achieves per ``start_pace``.

    python -m training.oracle_stats [--seeds train] [--num-seeds N] \
        [--max-steps N] [--max-pace N] [--export-demos demos.jsonl]

Reports, for each start pace: mean score at the step cap, how long the oracle takes
to clear the first maze tile against the shortest-path lower bound, the pace at the
end of the episode and the death rate. The highest start pace with no deaths is the
"highest sustainable pace". These ceilings choose the training ``start_pace`` and the
expert-ladder thresholds.
"""

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import IO, Any, Dict, List, Optional, Sequence

from infinite_maze.core.simulation import Simulation

from . import config
from .observation import build_observation, flatten_observation
from .oracle import MazeGeometry, OraclePolicy
from .reward import RewardFunction
from .stepping import repeat_action


@dataclass
class OracleEpisode:
    """One oracle episode on one seed."""

    seed: int
    score: int
    steps: int
    terminated: bool
    final_pace: int
    tile_clear_step: Optional[int]  # tick the first tile was cleared, if it was
    lower_bound_steps: Optional[int]  # shortest path through the tile, in ticks


def run_episode(
    sim: Simulation,
    policy: OraclePolicy,
    frame_skip: int,
    demos: Optional[IO[str]] = None,
    gridSize: int = config.OBS_GRID_SIZE,
    cellPx: int = config.OBS_GRID_CELL_PX,
) -> OracleEpisode:
    """Play one episode, tracking first-tile clearing (and exporting demos)."""
    policy.reset(sim.getSeed())
    geometry = MazeGeometry()
    reward = RewardFunction()
    tileEnd = geometry.originX + geometry.period
    cells = policy.cells_to_column(sim, geometry.cols)
    lowerBound = None if cells is None else cells * geometry.cell
    cleared: Optional[int] = None
    terminated = False
    while True:
        action = policy.act_privileged(sim)
        if demos is not None:
            record = {
                "seed": sim.getSeed(),
                "tick": sim.getTicks(),
                "obs": flatten_observation(build_observation(sim, gridSize, cellPx)),
                "action": action,
            }
            demos.write(json.dumps(record) + "\n")
        outcome = repeat_action(sim, action, reward, frame_skip)
        worldX = sim.player.getX() + sim.getScrollOffset()
        if cleared is None and worldX >= tileEnd:
            cleared = sim.getSteps()
        if outcome.terminated or outcome.truncated:
            terminated = outcome.terminated
            break
    return OracleEpisode(
        seed=sim.getSeed(),
        score=sim.game.getScore(),
        steps=sim.getSteps(),
        terminated=terminated,
        final_pace=sim.game.getPace(),
        tile_clear_step=cleared,
        lower_bound_steps=lowerBound,
    )


def stats_for_pace(
    seeds: Sequence[int],
    start_pace: int,
    max_steps: int,
    frame_skip: int,
    demos: Optional[IO[str]] = None,
    gridSize: int = config.OBS_GRID_SIZE,
    cellPx: int = config.OBS_GRID_CELL_PX,
) -> Dict[str, Any]:
    """Aggregate oracle results over ``seeds`` at one start pace."""
    policy = OraclePolicy()
    sim = Simulation(seed=seeds[0], start_pace=start_pace, max_steps=max_steps)
    episodes: List[OracleEpisode] = []
    for index, seed in enumerate(seeds):
        if index > 0:
            sim.reset(seed)
        episodes.append(run_episode(sim, policy, frame_skip, demos, gridSize, cellPx))

    clearTimes = [e.tile_clear_step for e in episodes if e.tile_clear_step is not None]
    ratios = [
        e.tile_clear_step / e.lower_bound_steps
        for e in episodes
        if e.tile_clear_step is not None and e.lower_bound_steps
    ]
    return {
        "start_pace": start_pace,
        "mean_score": statistics.fmean(e.score for e in episodes),
        "mean_steps": statistics.fmean(e.steps for e in episodes),
        "death_rate": sum(e.terminated for e in episodes) / len(episodes),
        "mean_final_pace": statistics.fmean(e.final_pace for e in episodes),
        "tile_cleared_rate": len(clearTimes) / len(episodes),
        "mean_tile_clear_step": statistics.fmean(clearTimes) if clearTimes else None,
        "mean_clear_vs_lower_bound": statistics.fmean(ratios) if ratios else None,
        "unsolved_seeds": [e.seed for e in episodes if e.tile_clear_step is None],
        "episodes": [asdict(e) for e in episodes],
    }


def format_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Ceiling table, one row per start pace."""

    def cell(value: Optional[float], spec: str) -> str:
        return "-" if value is None else format(value, spec)

    header = (
        f"{'pace':>5}{'mean score':>12}{'mean steps':>12}{'death':>8}"
        f"{'final pace':>12}{'tile clr':>10}{'clr steps':>11}{'vs bound':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['start_pace']:>5}{r['mean_score']:>12.1f}{r['mean_steps']:>12.0f}"
            f"{r['death_rate']:>8.0%}{r['mean_final_pace']:>12.1f}"
            f"{r['tile_cleared_rate']:>10.0%}"
            f"{cell(r['mean_tile_clear_step'], '.0f'):>11}"
            f"{cell(r['mean_clear_vs_lower_bound'], '.2f'):>10}"
        )
    sustainable = [r["start_pace"] for r in rows if r["death_rate"] == 0]
    lines.append(
        "highest sustainable start pace: "
        + (str(max(sustainable)) if sustainable else "none")
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m training.oracle_stats",
        description="Oracle ceilings per start pace on a fixed seed set.",
    )
    parser.add_argument(
        "--seeds", default="train", choices=sorted(config.SEED_SETS), help="seed set"
    )
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=config.OBS_GRID_SIZE)
    parser.add_argument("--cell-px", type=int, default=config.OBS_GRID_CELL_PX)
    parser.add_argument("--min-pace", type=int, default=0)
    parser.add_argument("--max-pace", type=int, default=config.ORACLE_STATS_MAX_PACE)
    parser.add_argument("--out", default=None, help="write the full stats as JSON")
    parser.add_argument(
        "--export-demos",
        default=None,
        help="write (flattened observation, action) pairs as JSONL",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    seeds = list(config.SEED_SETS[args.seeds])
    if args.num_seeds is not None:
        seeds = seeds[: args.num_seeds]
    if args.frame_skip < 1 or args.max_steps < 1:
        raise SystemExit("--frame-skip and --max-steps must be positive")

    demos = open(args.export_demos, "w") if args.export_demos else None
    try:
        rows = []
        for pace in range(args.min_pace, args.max_pace + 1):
            rows.append(
                stats_for_pace(
                    seeds,
                    pace,
                    args.max_steps,
                    args.frame_skip,
                    demos,
                    args.grid_size,
                    args.cell_px,
                )
            )
            print(f"start_pace {pace} done", file=sys.stderr, flush=True)
    finally:
        if demos is not None:
            demos.close()

    print(format_table(rows))
    for row in rows:
        if row["unsolved_seeds"]:
            print(
                f"pace {row['start_pace']}: tile not cleared on seeds "
                f"{row['unsolved_seeds']}"
            )
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "seeds": args.seeds,
                    "max_steps": args.max_steps,
                    "frame_skip": args.frame_skip,
                    "rows": rows,
                },
                handle,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
