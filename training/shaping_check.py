"""Reward-ordering gate: do shaped training returns rank the baselines correctly?

    python -m training.shaping_check --num-seeds 20 --weight 44 --scale 0.05

Plays random, right, heuristic and oracle with the same shaped reward the training
environment uses (``PotentialShaper`` + ``repeat_action``; stdlib only) and checks
the order oracle > heuristic > right > random. If the oracle does not clearly lead,
the shaping is wrong: fix it before training.

The ranking metric is the **discounted** training return, one discount per decision
with ``gamma = shaping.gamma``, because that is what PPO maximises. The undiscounted
sum is printed for information only: with ``gamma < 1`` a potential ``-d(cell)``
pays ``weight * (1 - gamma) * d`` per decision just for being far from the goal, so
the undiscounted sum rewards time spent in the maze (it ranked the heuristic above
the oracle). That term is exactly compensated in the discounted return (potential
shaping is policy invariant under discounting). The discounted return is shifted by
the start potential, a constant per seed, so it reads as "progress made".
"""

import argparse
import statistics
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence

from infinite_maze.core.simulation import Simulation

from . import config
from .evaluate import build_policy
from .observation import build_observation
from .policies import Policy, PrivilegedPolicy
from .reward import RewardConfig, RewardFunction
from .shaping import PotentialShaper, ShapingConfig
from .stepping import repeat_action

GATE_POLICIES = ("random", "right", "heuristic", "oracle")
EXPECTED_ORDER = ("oracle", "heuristic", "right", "random")


class GateReturn(NamedTuple):
    """Training return of one episode (shaping and scale included)."""

    undiscounted: float
    discounted: float  # per-decision discount, shifted by the start potential


def shaped_return(
    policy: Policy,
    sim: Simulation,
    reward: RewardFunction,
    shaper: PotentialShaper,
    frame_skip: int,
    reward_scale: float,
) -> GateReturn:
    """Training return of one episode on ``sim``."""
    policy.reset(sim.getSeed())
    shaper.reset(sim)
    gamma = shaper.config.gamma
    undiscounted = 0.0
    discounted = 0.0
    discount = 1.0
    while True:
        if isinstance(policy, PrivilegedPolicy):
            action = policy.act_privileged(sim)
        else:
            action = policy.act(build_observation(sim))
        outcome = repeat_action(sim, action, reward, frame_skip)
        shaping = shaper.step(sim, outcome.terminated)
        step = (outcome.reward + shaping) * reward_scale
        undiscounted += step
        discounted += discount * step
        discount *= gamma
        if outcome.terminated or outcome.truncated:
            shift = shaper.config.weight * shaper.start * reward_scale
            return GateReturn(undiscounted, discounted + shift)


def mean_training_returns(
    seeds: Sequence[int],
    shaping: ShapingConfig,
    reward_scale: float,
    max_steps: int = config.EVAL_MAX_STEPS,
    start_pace: int = config.EVAL_START_PACE,
    frame_skip: int = config.DEFAULT_FRAME_SKIP,
    reward_config: Optional[RewardConfig] = None,
    policies: Sequence[str] = GATE_POLICIES,
) -> Dict[str, GateReturn]:
    """Mean shaped training return per policy over ``seeds``."""
    rewardConfig = reward_config if reward_config is not None else RewardConfig()
    rewardConfig.validate(max_steps)
    reward = RewardFunction(rewardConfig)
    results: Dict[str, GateReturn] = {}
    for name in policies:
        policy = build_policy(name)
        shaper = PotentialShaper(shaping, max_steps)
        returns: List[GateReturn] = []
        sim = Simulation(seed=seeds[0], start_pace=start_pace, max_steps=max_steps)
        for index, seed in enumerate(seeds):
            if index > 0:
                sim.reset(seed)
            returns.append(
                shaped_return(policy, sim, reward, shaper, frame_skip, reward_scale)
            )
        results[name] = GateReturn(
            statistics.fmean(r.undiscounted for r in returns),
            statistics.fmean(r.discounted for r in returns),
        )
    return results


def ordering_holds(results: Dict[str, GateReturn]) -> bool:
    """Whether the discounted means strictly follow ``EXPECTED_ORDER`` (best first)."""
    values = [results[name].discounted for name in EXPECTED_ORDER]
    return all(a > b for a, b in zip(values, values[1:]))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m training.shaping_check",
        description="Check that shaped training returns rank the baselines.",
    )
    parser.add_argument("--seeds", default="train", choices=sorted(config.SEED_SETS))
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--weight", type=float, default=44.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--start-pace", type=int, default=config.EVAL_START_PACE)
    parser.add_argument("--frame-skip", type=int, default=config.DEFAULT_FRAME_SKIP)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point; exit code 1 when the expected ordering does not hold."""
    args = build_parser().parse_args(argv)
    seeds = list(config.SEED_SETS[args.seeds])[: args.num_seeds]
    results = mean_training_returns(
        seeds,
        ShapingConfig(args.weight, args.gamma),
        args.scale,
        args.max_steps,
        args.start_pace,
        args.frame_skip,
    )
    print(f"{'policy':<12}{'discounted (gate)':>20}{'undiscounted':>16}")
    for name in EXPECTED_ORDER:
        r = results[name]
        print(f"{name:<12}{r.discounted:>20.2f}{r.undiscounted:>16.2f}")
    held = ordering_holds(results)
    print("ordering oracle > heuristic > right > random:", "OK" if held else "FAILED")
    return 0 if held else 1


if __name__ == "__main__":
    sys.exit(main())
