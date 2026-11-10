"""Observation probe: can a small MLP solve the mazes from this observation?

    python -m training.probe --grid-size 21 [--no-extras] [--hidden 64 64]
    python -m training.probe --sweep            # plan 4 R2: windows x features

Behaviour cloning plus DAgger on oracle labels, as a minutes-long proxy for "is the
answer in this observation?" (plan 4 section 10). The oracle is only a teacher
here: it labels the states visited on the *fit* seeds (train_extra 3000-3399).
Train seeds 0-19 are held out for action accuracy, and the validation seeds are
only played through the shared ``evaluate()``, never labelled. Observations are the
ordinary non-privileged ones (wall raster, rays, scalars); only the raster size
varies, and every size stays inside the rendered 640 x 480 screen.

Round 0 is plain cloning (the oracle acts). Each later round lets the current
learner act (the oracle acts with probability ``beta``), labels every visited state
with the oracle and refits on the aggregated data. Plain cloning only sees
oracle-visited states, so without DAgger a probe fails off-distribution whatever
the observation shows.

*Decision states* are states where the oracle's next cell is not straight ahead
(it turns, climbs, descends or backs up). Accuracy there is the number that says
whether the observation disambiguates the maze; overall accuracy is dominated by
"keep going right".
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from infinite_maze.core.simulation import Simulation
from infinite_maze.utils.config import config as game_config

from . import config
from .evaluate import EvalReport, build_policy, evaluate
from .observation import (
    Observation,
    build_observation,
    flatten_observation,
    observation_size,
)
from .oracle import OraclePolicy
from .policies import Policy
from .windowed import WindowedPlanner
from .reward import RewardFunction
from .stepping import repeat_action

NUM_ACTIONS = len(game_config.MOVEMENT_CONSTANTS)


@dataclass(frozen=True)
class ProbeConfig:
    """Everything that defines one probe run."""

    grid_size: int = config.OBS_GRID_SIZE
    cell_px: int = config.OBS_GRID_CELL_PX
    extras: bool = True  # include the rays and scalars next to the raster
    arch: str = "mlp"  # "mlp" (the PPO policy shape) or "cnn" (inductive-bias control)
    hidden: Tuple[int, ...] = config.PROBE_HIDDEN
    betas: Tuple[float, ...] = config.PROBE_BETAS
    epochs: int = config.PROBE_EPOCHS
    batch_size: int = config.PROBE_BATCH
    learning_rate: float = config.PROBE_LEARNING_RATE
    stride: int = config.PROBE_STRIDE
    max_steps: int = config.EVAL_MAX_STEPS_PROBE
    frame_skip: int = config.DEFAULT_FRAME_SKIP
    start_pace: int = config.EVAL_START_PACE
    seed: int = 0

    def input_size(self) -> int:
        """Length of the vector the learner reads."""
        size = observation_size(self.grid_size)
        return size if self.extras else self.grid_size * self.grid_size

    def tag(self) -> str:
        """Short name for output files."""
        parts = [self.arch, f"g{self.grid_size}", "ext" if self.extras else "grid"]
        parts.append("h" + "x".join(str(h) for h in self.hidden))
        return "_".join(parts) + f"_s{self.seed}"


def check_probe_seeds(fit: Sequence[int], holdout: Sequence[int]) -> None:
    """Refuse anything but training-set seeds for labelling, and overlapping splits."""
    allowed = set(config.TRAIN_SEEDS) | set(config.TRAIN_EXTRA_SEEDS)
    for label, seeds in (("fit", fit), ("holdout", holdout)):
        if not seeds or not allowed.issuperset(seeds):
            raise ValueError(
                f"{label} seeds must be non-empty seeds of the train or "
                f"train_extra sets (never validation or test)"
            )
    if set(fit) & set(holdout):
        raise ValueError("fit and holdout seeds must not overlap")


def observation_vector(obs: Observation, cfg: ProbeConfig) -> List[float]:
    """The learner's input: the shared flattening, raster only when no extras."""
    flat = flatten_observation(obs)
    return flat if cfg.extras else flat[: cfg.grid_size * cfg.grid_size]


class ConvNet(nn.Module):
    """Raster through strided convolutions, extras joined before the dense head."""

    def __init__(self, cfg: ProbeConfig) -> None:
        super().__init__()
        self.size = cfg.grid_size
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten(),
        )
        reduced = (((self.size + 1) // 2) + 1) // 2
        joined = 32 * reduced * reduced + cfg.input_size() - self.size * self.size
        layers: List[Any] = []
        width = joined
        for units in cfg.hidden:
            layers += [nn.Linear(width, units), nn.Tanh()]
            width = units
        layers.append(nn.Linear(width, NUM_ACTIONS))
        self.head = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        raster = x[:, : self.size * self.size].reshape(-1, 1, self.size, self.size)
        features = self.convs(raster)
        extras = x[:, self.size * self.size :]
        return self.head(torch.cat([features, extras], dim=1))


def build_model(cfg: ProbeConfig) -> Any:
    """A fresh network over the flat observation vector."""
    if cfg.arch == "cnn":
        return ConvNet(cfg)
    if cfg.arch != "mlp":
        raise ValueError(f"unknown arch {cfg.arch!r}")
    sizes = [cfg.input_size(), *cfg.hidden, NUM_ACTIONS]
    layers: List[Any] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Net:
    """A trained network evaluated on numpy inputs (one thread, no gradients)."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def logits(self, x: Any) -> Any:
        single = np.ndim(x) == 1
        with torch.no_grad():
            tensor = torch.from_numpy(np.atleast_2d(np.asarray(x, np.float32)))
            out = self.model(tensor).numpy()
        return out[0] if single else out

    def predict(self, x: Any) -> Any:
        if np.ndim(x) == 1:
            return int(np.argmax(self.logits(x)))
        return np.concatenate(
            [
                np.argmax(self.logits(x[i : i + 8192]), axis=-1)
                for i in range(0, len(x), 8192)
            ]
        )

    def probabilities(self, x: Any) -> Any:
        z = self.logits(x)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)


class ProbePolicy(Policy):
    """The probe network as a ``Policy``: deterministic, observation-only."""

    name = "probe"

    def __init__(self, net: Net, cfg: ProbeConfig) -> None:
        self.net = net
        self.cfg = cfg
        self.grid_size = cfg.grid_size
        self.cell_px = cfg.cell_px

    def act(self, obs: Observation) -> int:
        x = np.asarray(observation_vector(obs, self.cfg), dtype=np.float32)
        return int(self.net.predict(x))


@dataclass
class Batch:
    """Labelled states from rollouts."""

    x: Any
    y: Any
    decision: Any  # bool per state: the oracle's next cell is not straight ahead

    @classmethod
    def empty(cls, width: int) -> "Batch":
        return cls(
            np.zeros((0, width), np.float32),
            np.zeros(0, np.int64),
            np.zeros(0, bool),
        )

    def extend(self, other: "Batch") -> "Batch":
        return Batch(
            np.concatenate([self.x, other.x]),
            np.concatenate([self.y, other.y]),
            np.concatenate([self.decision, other.decision]),
        )


@dataclass
class Rollout:
    """One rollout: which maze, who acts, how."""

    seed: int
    cfg: ProbeConfig
    net: Optional[Net]  # None: the oracle acts
    beta: float  # probability that the oracle acts instead of the learner
    mode: str  # "sample" or "greedy" for the learner
    rng_seed: int


def collect(task: Rollout) -> Batch:
    """Play one episode and label every ``stride``-th visited state with the oracle."""
    cfg = task.cfg
    torch.set_num_threads(1)
    sim = Simulation(seed=task.seed, start_pace=cfg.start_pace, max_steps=cfg.max_steps)
    oracle = OraclePolicy()
    oracle.reset(task.seed)
    reward = RewardFunction()
    rng = np.random.default_rng(task.rng_seed)
    xs: List[List[float]] = []
    ys: List[int] = []
    flags: List[bool] = []
    index = 0
    while True:
        label = oracle.act_privileged(sim)
        geometry = oracle.geometry
        assert geometry is not None
        cell = geometry.cell_of(sim)
        nxt = oracle.next_cell(geometry, cell)
        obs = build_observation(sim, cfg.grid_size, cfg.cell_px)
        vector = observation_vector(obs, cfg)
        if index % cfg.stride == 0:
            xs.append(vector)
            ys.append(label)
            flags.append(nxt is not None and nxt != (cell[0] + 1, cell[1]))
        index += 1

        if task.net is None or rng.random() < task.beta:
            action = label
        elif task.mode == "greedy":
            action = int(task.net.predict(np.asarray(vector, dtype=np.float32)))
        else:
            probs = task.net.probabilities(np.asarray(vector, dtype=np.float32))
            action = int(rng.choice(NUM_ACTIONS, p=probs / probs.sum()))
        outcome = repeat_action(sim, action, reward, cfg.frame_skip)
        if outcome.terminated or outcome.truncated:
            break
    return Batch(
        np.asarray(xs, np.float32), np.asarray(ys, np.int64), np.asarray(flags, bool)
    )


def collect_many(tasks: Sequence[Rollout], workers: int) -> Batch:
    """Run rollouts (in order, so results do not depend on ``workers``)."""
    if workers > 1:
        with Pool(workers) as pool:
            batches = pool.map(collect, tasks, chunksize=1)
    else:
        batches = [collect(task) for task in tasks]
    width = tasks[0].cfg.input_size()
    merged = Batch.empty(width)
    for batch in batches:
        merged = merged.extend(batch)
    return merged


def fit(
    net: Optional[Net], data: Batch, cfg: ProbeConfig, round_index: int
) -> Tuple[Net, float]:
    """Cross-entropy fit continuing from ``net``; returns the net and last loss."""
    torch.set_num_threads(config.PROBE_FIT_THREADS)
    torch.manual_seed(cfg.seed * 1009 + round_index)
    model = net.model if net is not None else build_model(cfg)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    x = torch.from_numpy(data.x)
    y = torch.from_numpy(data.y)
    loss = torch.tensor(0.0)
    for _ in range(cfg.epochs):
        order = torch.randperm(len(y))
        for start in range(0, len(y), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            optimiser.zero_grad()
            loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            optimiser.step()
    model.eval()
    return Net(model), float(loss.item())


def accuracy(net: Net, data: Batch) -> Dict[str, float]:
    """Overall and decision-state accuracy, plus the majority-class baseline."""
    if len(data.y) == 0:
        return {}
    hit = net.predict(data.x) == data.y
    result: Dict[str, float] = {
        "states": float(len(data.y)),
        "accuracy": float(hit.mean()),
        "decision_states": float(data.decision.sum()),
    }
    if data.decision.any():
        labels = data.y[data.decision]
        result["decision_accuracy"] = float(hit[data.decision].mean())
        result["decision_majority"] = float(np.bincount(labels).max() / len(labels))
    return result


def parallel_evaluate(
    policy: Policy, seeds: Sequence[int], cfg: ProbeConfig, workers: int
) -> EvalReport:
    """``evaluate()`` over chunks of seeds in parallel, merged per seed."""
    chunks = [list(seeds[i::workers]) for i in range(workers) if seeds[i::workers]]
    jobs = [(policy, chunk, cfg) for chunk in chunks]
    with Pool(len(chunks)) as pool:
        reports = pool.starmap(_evaluate_chunk, jobs)
    episodes = sorted(
        (e for report in reports for e in report.episodes),
        key=lambda e: list(seeds).index(e.seed),
    )
    first = reports[0]
    return EvalReport(
        first.policy,
        first.max_steps,
        first.start_pace,
        first.reward,
        episodes,
        first.frame_skip,
    )


def _evaluate_chunk(policy: Policy, seeds: List[int], cfg: ProbeConfig) -> EvalReport:
    return evaluate(
        policy,
        seeds,
        max_steps=cfg.max_steps,
        start_pace=cfg.start_pace,
        frame_skip=cfg.frame_skip,
    )


@dataclass
class ProbeResult:
    """Outcome of one probe run (JSON-friendly through ``to_dict``)."""

    cfg: ProbeConfig
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    fit: Dict[str, float] = field(default_factory=dict)
    validation: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "rounds": self.rounds,
            "fit_seeds": self.fit,
            "validation": self.validation,
        }


def run_probe(
    cfg: ProbeConfig,
    workers: int = config.PROBE_WORKERS,
    fit_seeds: Optional[Sequence[int]] = None,
    holdout_seeds: Optional[Sequence[int]] = None,
    eval_fit_seeds: Optional[Sequence[int]] = None,
    eval_validation_seeds: Optional[Sequence[int]] = None,
    log: bool = True,
) -> Tuple[ProbeResult, ProbePolicy]:
    """Clone, then DAgger; evaluate the final network with the shared harness."""
    fitSeeds = list(config.PROBE_FIT_SEEDS if fit_seeds is None else fit_seeds)
    holdSeeds = list(
        config.PROBE_HOLDOUT_SEEDS if holdout_seeds is None else holdout_seeds
    )
    check_probe_seeds(fitSeeds, holdSeeds)
    fitEval = list(
        config.PROBE_FIT_EVAL_SEEDS if eval_fit_seeds is None else eval_fit_seeds
    )
    validation = list(
        config.VALIDATION_SEEDS
        if eval_validation_seeds is None
        else eval_validation_seeds
    )

    result = ProbeResult(cfg)
    data = Batch.empty(cfg.input_size())
    net: Optional[Net] = None
    for round_index, beta in enumerate(cfg.betas):
        base = cfg.seed * 1_000_003 + round_index * 10_007
        tasks = [Rollout(s, cfg, net, beta, "sample", base + s) for s in fitSeeds]
        data = data.extend(collect_many(tasks, workers))
        net, loss = fit(net, data, cfg, round_index)

        holdTasks = [Rollout(s, cfg, net, 0.0, "greedy", base + s) for s in holdSeeds]
        row = {
            "round": round_index,
            "beta": beta,
            "samples": len(data.y),
            "loss": loss,
            "fit": accuracy(net, data),
            "holdout": accuracy(net, collect_many(holdTasks, workers)),
        }
        result.rounds.append(row)
        if log:
            hold: Dict[str, float] = row["holdout"]  # type: ignore[assignment]
            print(
                f"[{cfg.tag()}] round {round_index}: {row['samples']} states, "
                f"loss {loss:.3f}, holdout acc {hold.get('accuracy', 0):.3f}, "
                f"decision acc {hold.get('decision_accuracy', 0):.3f} "
                f"(majority {hold.get('decision_majority', 0):.3f})",
                flush=True,
            )

    assert net is not None
    policy = ProbePolicy(net, cfg)
    result.fit = parallel_evaluate(policy, fitEval, cfg, workers).summary()
    result.validation = parallel_evaluate(policy, validation, cfg, workers).summary()
    return result, policy


def reference_scores(cfg: ProbeConfig, workers: int) -> Dict[str, Dict[str, float]]:
    """Heuristic and oracle on the validation seeds, same settings as the probes."""
    seeds = list(config.VALIDATION_SEEDS)
    return {
        name: parallel_evaluate(build_policy(name), seeds, cfg, workers).summary()
        for name in ("heuristic", "oracle")
    }


def g1_reading(scores: Dict[str, float]) -> str:
    """Plain-language reading of plan 4's gate G1 from {tag: validation score}."""
    best = max(scores.values())
    if best >= config.PROBE_G1_PASS_SCORE:
        return "pass: a window reaches the pass score (answer is in a wider window)"
    if best <= config.PROBE_G1_FAIL_SCORE:
        return "fail: no window beats the fail score (memoryless views insufficient)"
    return "inconclusive: between the fail and pass scores"


def sweep_configs(seed: int) -> List[ProbeConfig]:
    """Windows x with/without extras, plus the widest window with a bigger net."""
    configs = [
        ProbeConfig(grid_size=size, extras=extras, seed=seed)
        for size in config.PROBE_WINDOWS
        for extras in (True, False)
    ]
    configs.append(
        ProbeConfig(
            grid_size=config.PROBE_WINDOWS[-1],
            hidden=config.PROBE_CONTROL_HIDDEN,
            seed=seed,
        )
    )
    return configs


def horizon_sweep(
    horizons: Sequence[int], workers: int, cfg: Optional[ProbeConfig] = None
) -> Dict[int, Dict[str, float]]:
    """Information ceiling: a perfect planner that only looks ``h`` columns ahead.

    ``OraclePolicy(lookahead_cols=h)`` knows every wall but plans only to the column
    ``h`` ahead, the way a learner limited to a window of that reach could at best.
    It needs no fitting, so it separates "the window lacks the information" from "the
    network cannot learn it". A raster of ``g`` cells of ``cell_px`` reaches
    ``g * cell_px / 2 / maze_cell`` columns ahead of the player.
    """
    cfg = cfg if cfg is not None else ProbeConfig()
    seeds = list(config.VALIDATION_SEEDS)
    return {
        h: parallel_evaluate(
            OraclePolicy(lookahead_cols=h), seeds, cfg, workers
        ).summary()
        for h in horizons
    }


def window_sweep(
    grid_sizes: Sequence[int], workers: int, cfg: Optional[ProbeConfig] = None
) -> Dict[int, Dict[str, float]]:
    """Memoryless-window ceiling: ``WindowedPlanner`` per raster size (validation)."""
    cfg = cfg if cfg is not None else ProbeConfig()
    seeds = list(config.VALIDATION_SEEDS)
    return {
        g: parallel_evaluate(
            WindowedPlanner(g, cfg.cell_px), seeds, cfg, workers
        ).summary()
        for g in grid_sizes
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m training.probe",
        description="Behaviour cloning + DAgger probe of an observation window.",
    )
    parser.add_argument("--sweep", action="store_true", help="plan 4 R2 sweep")
    parser.add_argument(
        "--window",
        type=int,
        nargs="+",
        default=None,
        help="memoryless windowed-planner ceiling per raster size instead of a probe",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        nargs="+",
        default=None,
        help="oracle planning-horizon ceiling (columns ahead) instead of a probe",
    )
    parser.add_argument("--grid-size", type=int, default=config.OBS_GRID_SIZE)
    parser.add_argument("--cell-px", type=int, default=config.OBS_GRID_CELL_PX)
    parser.add_argument("--no-extras", action="store_true", help="raster only")
    parser.add_argument("--arch", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--hidden", type=int, nargs="+", default=config.PROBE_HIDDEN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=config.PROBE_WORKERS)
    parser.add_argument(
        "--out-dir", default=str(config.RUNS_DIR / "probe"), help="JSON results dir"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.window:
        for size, summary in window_sweep(args.window, args.workers).items():
            print(
                f"window {size:2d} cells: validation score "
                f"{summary['mean_score']:.1f}, death {summary['death_rate']:.0%}"
            )
        return 0
    if args.horizon:
        for horizon, summary in horizon_sweep(args.horizon, args.workers).items():
            print(
                f"horizon {horizon:2d} columns: validation score "
                f"{summary['mean_score']:.1f}, death {summary['death_rate']:.0%}"
            )
        return 0
    outDir = Path(args.out_dir)
    outDir.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        configs = sweep_configs(args.seed)
    else:
        configs = [
            ProbeConfig(
                grid_size=args.grid_size,
                cell_px=args.cell_px,
                extras=not args.no_extras,
                arch=args.arch,
                hidden=tuple(args.hidden),
                seed=args.seed,
            )
        ]
    reference = reference_scores(configs[0], args.workers)
    print("reference (validation):", json.dumps(reference), flush=True)

    scores: Dict[str, float] = {}
    for cfg in configs:
        result, policy = run_probe(cfg, args.workers)
        scores[cfg.tag()] = result.validation["mean_score"]
        payload = result.to_dict()
        payload["reference"] = reference
        with open(outDir / f"{cfg.tag()}.json", "w") as handle:
            json.dump(payload, handle, indent=2)
        torch.save(policy.net.model.state_dict(), outDir / f"{cfg.tag()}.pt")
        print(
            f"[{cfg.tag()}] validation score {result.validation['mean_score']:.1f} "
            f"death {result.validation['death_rate']:.0%} | fit-20 score "
            f"{result.fit['mean_score']:.1f}",
            flush=True,
        )
    print("G1:", g1_reading(scores))
    return 0


if __name__ == "__main__":
    sys.exit(main())
