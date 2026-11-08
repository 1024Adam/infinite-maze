"""Fixed-seed evaluation of policies on ``Simulation`` (no rendering, no sleeping).

    python -m training.evaluate --policy all --seeds validation \
        --out training/runs/baselines.json
"""

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from infinite_maze.core.replay import ReplayRecorder, saveReplay
from infinite_maze.core.simulation import Simulation
from infinite_maze.utils.config import config as game_config

from . import config
from .observation import build_observation
from .policies import Policy, make_policy
from .reward import RewardConfig, RewardFunction, load_reward_config

BASELINES = ("random", "right", "heuristic")


@dataclass
class EpisodeResult:
    """Outcome of one policy episode on one seed."""

    seed: int
    score: int
    steps: int
    total_return: float
    terminated: bool
    truncated: bool

    def to_dict(self) -> Dict[str, Any]:
        """JSON form (``total_return`` is stored under the key ``return``)."""
        values = asdict(self)
        values["return"] = values.pop("total_return")
        return values


@dataclass
class EvalReport:
    """Per-seed results plus summary statistics for one policy."""

    policy: str
    max_steps: int
    start_pace: int
    reward: RewardConfig
    episodes: List[EpisodeResult]

    def returns(self) -> List[float]:
        """Return of every episode, in seed order."""
        return [episode.total_return for episode in self.episodes]

    def summary(self) -> Dict[str, float]:
        """Mean/median/min/max return, mean score and steps, and death rate."""
        returns = self.returns()
        count = len(self.episodes)
        return {
            "mean_return": statistics.fmean(returns),
            "median_return": statistics.median(returns),
            "min_return": min(returns),
            "max_return": max(returns),
            "mean_score": statistics.fmean(e.score for e in self.episodes),
            "mean_steps": statistics.fmean(e.steps for e in self.episodes),
            "death_rate": sum(e.terminated for e in self.episodes) / count,
        }

    def worst_seed(self) -> int:
        """Seed of the episode with the lowest return."""
        return min(self.episodes, key=lambda e: e.total_return).seed

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly form of the whole report."""
        return {
            "policy": self.policy,
            "max_steps": self.max_steps,
            "start_pace": self.start_pace,
            "reward": asdict(self.reward),
            "summary": self.summary(),
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def _run_episode(
    policy: Policy,
    sim: Simulation,
    reward: RewardFunction,
    recorder: Optional[ReplayRecorder] = None,
) -> EpisodeResult:
    """Play ``sim`` with ``policy`` until it terminates or is truncated."""
    policy.reset(sim.getSeed())
    total = 0.0
    while True:
        action = policy.act(build_observation(sim))
        intent = game_config.ACTION_TO_INTENT[action]
        result = sim.step(intent)
        if recorder is not None:
            recorder.record(intent)
        total += reward(result).total
        if result.terminated or result.truncated:
            return EpisodeResult(
                seed=sim.getSeed(),
                score=sim.game.getScore(),
                steps=sim.getSteps(),
                total_return=total,
                terminated=result.terminated,
                truncated=result.truncated,
            )


def evaluate(
    policy: Policy,
    seeds: Sequence[int],
    max_steps: int = config.EVAL_MAX_STEPS,
    start_pace: int = config.EVAL_START_PACE,
    reward: Optional[RewardConfig] = None,
) -> EvalReport:
    """Run ``policy`` once per seed and report the results.

    Deterministic for a given policy, seeds and settings.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive so every episode ends")
    rewardConfig = reward if reward is not None else RewardConfig()
    rewardConfig.validate(max_steps)
    rewardFunction = RewardFunction(rewardConfig)

    sim = Simulation(seed=seeds[0], start_pace=start_pace, max_steps=max_steps)
    episodes = []
    for index, seed in enumerate(seeds):
        if index > 0:
            sim.reset(seed)
        episodes.append(_run_episode(policy, sim, rewardFunction))
    return EvalReport(policy.name, max_steps, start_pace, rewardConfig, episodes)


def record_replay(
    policy: Policy,
    seed: int,
    max_steps: int = config.EVAL_MAX_STEPS,
    start_pace: int = config.EVAL_START_PACE,
    reward: Optional[RewardConfig] = None,
) -> Dict[str, Any]:
    """Re-run one episode and return it as a replay (watch it with the overlay)."""
    sim = Simulation(seed=seed, start_pace=start_pace, max_steps=max_steps)
    recorder = ReplayRecorder(seed, start_pace)
    _run_episode(policy, sim, RewardFunction(reward), recorder)
    return recorder.toReplay(sim.getState())


def format_table(reports: Sequence[EvalReport]) -> str:
    """Human-readable comparison of several reports."""
    header = (
        f"{'policy':<18}{'mean return':>12}{'median':>10}{'mean score':>12}"
        f"{'mean steps':>12}{'death rate':>12}"
    )
    rows = [header, "-" * len(header)]
    for report in reports:
        s = report.summary()
        rows.append(
            f"{report.policy:<18}{s['mean_return']:>12.1f}{s['median_return']:>10.1f}"
            f"{s['mean_score']:>12.1f}{s['mean_steps']:>12.0f}"
            f"{s['death_rate']:>12.0%}"
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m training.evaluate",
        description="Evaluate baseline policies on fixed seed sets.",
    )
    parser.add_argument(
        "--policy",
        default="all",
        choices=list(BASELINES) + ["heuristic-raster", "all"],
        help="'all' runs random, right and heuristic",
    )
    parser.add_argument(
        "--seeds", default="validation", choices=sorted(config.SEED_SETS)
    )
    parser.add_argument(
        "--num-seeds", type=int, default=None, help="only use the first N seeds"
    )
    parser.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS)
    parser.add_argument("--start-pace", type=int, default=config.EVAL_START_PACE)
    parser.add_argument("--reward-config", default=None, help="JSON RewardConfig file")
    parser.add_argument("--out", default=None, help="write the reports as JSON")
    parser.add_argument(
        "--worst-replay-dir",
        default=None,
        help="write each policy's worst-seed episode as a replay in this directory",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point; prints a comparison table and returns the exit code."""
    args = build_parser().parse_args(argv)
    seeds = list(config.SEED_SETS[args.seeds])
    if args.num_seeds is not None:
        seeds = seeds[: args.num_seeds]
    reward = load_reward_config(args.reward_config) if args.reward_config else None
    names = list(BASELINES) if args.policy == "all" else [args.policy]

    reports = []
    for name in names:
        report = evaluate(
            make_policy(name), seeds, args.max_steps, args.start_pace, reward
        )
        reports.append(report)
        if args.worst_replay_dir:
            worst = report.worst_seed()
            path = Path(args.worst_replay_dir) / f"{name}-worst-seed{worst}.json"
            replay = record_replay(
                make_policy(name), worst, args.max_steps, args.start_pace, reward
            )
            saveReplay(replay, str(path))
            print(f"worst {name} episode (seed {worst}) written to {path}")

    print(format_table(reports))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"reports": [r.to_dict() for r in reports]}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
