"""Train a PPO agent from a committed experiment config (needs ``ml-train``).

    python -m training.train --config training/configs/<name>.json [--resume DIR]

Writes ``training/runs/<name>/``: config.json (copy), metadata.json, metrics.csv,
checkpoints/, best.zip, final.zip, eval.json and replays/ (best and worst eval
episodes, watchable with ``python -m training.overlay --replay``).
"""

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

# Dummy SDL drivers must be set before pygame is imported (also in subprocesses).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)

from infinite_maze.core.replay import saveReplay  # noqa: E402

from . import config  # noqa: E402
from .env import MazeEnv  # noqa: E402
from .evaluate import EvalReport, evaluate, record_replay  # noqa: E402
from .experiment import ExperimentConfig, load_experiment_config  # noqa: E402
from .model_policy import ModelPolicy  # noqa: E402

METRIC_COLUMNS = (
    "timesteps",
    "ep_return_mean",
    "ep_score_mean",
    "ep_length_mean",
    "policy_loss",
    "value_loss",
    "entropy_loss",
    "explained_variance",
)
LOSS_KEYS = {
    "policy_loss": "train/policy_gradient_loss",
    "value_loss": "train/value_loss",
    "entropy_loss": "train/entropy_loss",
    "explained_variance": "train/explained_variance",
}


def run_dir_for(cfg: ExperimentConfig, runs_dir: Path = config.RUNS_DIR) -> Path:
    """Directory a run of ``cfg`` writes to."""
    return runs_dir / cfg.name


def prepare_run_dir(run_dir: Path, cfg: ExperimentConfig, resume: bool) -> None:
    """Create the run layout; refuse to overwrite an existing run."""
    if resume:
        if not (run_dir / "config.json").is_file():
            raise FileNotFoundError(f"{run_dir} has no config.json to resume")
        return
    if run_dir.exists():
        raise FileExistsError(
            f"{run_dir} already exists: configs are never edited mid-run, so use a "
            f"new name, or --resume {run_dir}"
        )
    for sub in ("checkpoints", "replays"):
        (run_dir / sub).mkdir(parents=True)
    with open(run_dir / "config.json", "w") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)
        handle.write("\n")


def collect_metadata(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Git commit, library versions, Python, platform and seed of this run."""
    import gymnasium
    import stable_baselines3

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ("git",) + args, cwd=Path(__file__).parent, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "name": cfg.name,
        "seed": cfg.seed,
        "shaping": {"weight": cfg.shaping.weight, "gamma": cfg.shaping.gamma},
        "train_reward_scale": cfg.train_reward_scale,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain") not in ("", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
        },
    }


def make_training_env(cfg: ExperimentConfig, start_pace: int) -> VecEnv:
    """``n_envs`` environments (subprocesses when more than one), seeded from cfg."""
    factory = partial(
        MazeEnv,
        cfg.train_seed_list(),
        start_pace=start_pace,
        max_steps=cfg.max_steps,
        frame_skip=cfg.frame_skip,
        reward_config=cfg.reward,
        shaping=cfg.shaping,
        reward_scale=cfg.train_reward_scale,
    )
    return make_vec_env(
        factory,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv if cfg.n_envs > 1 else DummyVecEnv,
        monitor_kwargs={"info_keywords": ("score",)},
    )


def build_model(cfg: ExperimentConfig, env: VecEnv) -> PPO:
    """A fresh PPO model from the config."""
    ppo = cfg.ppo
    return PPO(
        ppo.policy,
        env,
        n_steps=ppo.n_steps,
        batch_size=ppo.batch_size,
        learning_rate=ppo.learning_rate,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        ent_coef=ppo.ent_coef,
        n_epochs=ppo.n_epochs,
        clip_range=ppo.clip_range,
        policy_kwargs={"net_arch": list(ppo.net_arch)},
        seed=cfg.seed,
        device="cpu",
        verbose=1,
    )


def evaluate_sets(
    cfg: ExperimentConfig, policy: ModelPolicy, eval_seeds: Dict[str, List[int]]
) -> Dict[str, EvalReport]:
    """Shared ``evaluate()`` on every eval set, with the training frame_skip."""
    return {
        label: evaluate(
            policy,
            seeds,
            max_steps=cfg.max_steps,
            start_pace=cfg.start_pace,
            reward=cfg.reward,
            frame_skip=cfg.frame_skip,
        )
        for label, seeds in eval_seeds.items()
    }


def write_replays(
    cfg: ExperimentConfig,
    policy: ModelPolicy,
    reports: Dict[str, EvalReport],
    out_dir: Path,
) -> None:
    """Write each eval set's best and worst episode as watchable replays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, report in reports.items():
        ordered = sorted(report.episodes, key=lambda e: e.total_return)
        for kind, episode in (("worst", ordered[0]), ("best", ordered[-1])):
            replay = record_replay(
                policy,
                episode.seed,
                cfg.max_steps,
                cfg.start_pace,
                cfg.reward,
                cfg.frame_skip,
            )
            name = f"{label}-{kind}-seed{episode.seed}.json"
            saveReplay(replay, str(out_dir / name))


def describe(entry: Dict[str, Any]) -> str:
    """One-line summary of an evaluation entry."""
    parts = []
    for label, summary in entry["sets"].items():
        parts.append(
            f"{label}: return {summary['mean_return']:.1f} "
            f"score {summary['mean_score']:.1f} death {summary['death_rate']:.0%}"
        )
    return " | ".join(parts)


class RunCallback(BaseCallback):
    """Writes metrics.csv each rollout and evaluates/checkpoints on a schedule."""

    def __init__(self, cfg: ExperimentConfig, run_dir: Path) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_dir = run_dir
        self.eval_seeds = cfg.eval_seed_lists()
        self.history: List[Dict[str, Any]] = []
        self.best_return = float("-inf")
        self.next_eval = 0
        path = run_dir / "eval.json"
        if path.is_file():
            with open(path) as handle:
                saved = json.load(handle)
            self.history = saved.get("history", [])
            self.best_return = saved.get("best_return", float("-inf"))

    def _schedule_next_eval(self) -> None:
        every = self.cfg.eval_every_timesteps
        self.next_eval = (self.model.num_timesteps // every + 1) * every

    def _on_training_start(self) -> None:
        self._schedule_next_eval()
        metrics = self.run_dir / "metrics.csv"
        if not metrics.is_file():
            with open(metrics, "w", newline="") as handle:
                csv.writer(handle).writerow(METRIC_COLUMNS)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buffer = self.model.ep_info_buffer
        row: Dict[str, Any] = {"timesteps": self.model.num_timesteps}
        if buffer:
            row["ep_return_mean"] = float(np.mean([e["r"] for e in buffer]))
            row["ep_length_mean"] = float(np.mean([e["l"] for e in buffer]))
            scores = [e["score"] for e in buffer if "score" in e]
            if scores:
                row["ep_score_mean"] = float(np.mean(scores))
        values = self.model.logger.name_to_value
        for column, key in LOSS_KEYS.items():
            if key in values:
                row[column] = float(values[key])
        with open(self.run_dir / "metrics.csv", "a", newline="") as handle:
            csv.writer(handle).writerow([row.get(c, "") for c in METRIC_COLUMNS])

        if self.model.num_timesteps >= self.next_eval:
            self.run_evaluation()
            self._schedule_next_eval()

    def run_evaluation(self, final: bool = False) -> Dict[str, Any]:
        """Evaluate every eval set, checkpoint, and keep the best model."""
        timesteps = self.model.num_timesteps
        policy = ModelPolicy(self.model)
        reports = evaluate_sets(self.cfg, policy, self.eval_seeds)
        entry: Dict[str, Any] = {
            "timesteps": timesteps,
            "sets": {label: r.summary() for label, r in reports.items()},
        }
        if final and self.history and self.history[-1]["timesteps"] == timesteps:
            self.history[-1] = entry  # the scheduled eval already ran at this step
        else:
            self.history.append(entry)

        self.model.save(self.run_dir / "checkpoints" / f"step_{timesteps}")
        self.model.save(self.run_dir / "checkpoints" / "latest")
        selected = reports.get(self.cfg.select_on)
        if selected is not None:
            mean_return = selected.summary()["mean_return"]
            if mean_return > self.best_return:
                self.best_return = mean_return
                self.model.save(self.run_dir / "best")
                replays = self.run_dir / "replays" / "best"
                write_replays(self.cfg, policy, reports, replays)
        if final:
            replays = self.run_dir / "replays" / "final"
            write_replays(self.cfg, policy, reports, replays)
        with open(self.run_dir / "eval.json", "w") as handle:
            json.dump(
                {"history": self.history, "best_return": self.best_return},
                handle,
                indent=2,
            )
        print(f"[eval @ {timesteps}] " + describe(entry), flush=True)
        return entry


def train(cfg: ExperimentConfig, run_dir: Path, resume: bool = False) -> PPO:
    """Run (or resume) training as ``cfg`` describes and return the final model.

    Seeds for numpy, torch, SB3 and the environments all derive from ``cfg.seed``.
    Known nondeterminism: a resumed run does not restore RNG state.
    """
    prepare_run_dir(run_dir, cfg, resume)
    if not resume:
        with open(run_dir / "metadata.json", "w") as handle:
            json.dump(collect_metadata(cfg), handle, indent=2)
            handle.write("\n")

    set_random_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(config.TORCH_THREADS)

    done = 0
    model: Optional[PPO] = None
    if resume:
        latest = run_dir / "checkpoints" / "latest.zip"
        if not latest.is_file():
            raise FileNotFoundError(f"{latest} missing: nothing to resume from")
        model = PPO.load(latest, device="cpu")
        done = model.num_timesteps

    callback = RunCallback(cfg, run_dir)
    stage_starts = [s.timesteps for s in cfg.curriculum if done < s.timesteps]
    segment_ends = [t for t in stage_starts if t < cfg.total_timesteps]
    segment_ends.append(cfg.total_timesteps)

    for end in segment_ends:
        if end <= done:
            continue
        env = make_training_env(cfg, cfg.pace_at(done))
        if model is None:
            model = build_model(cfg, env)
        else:
            model.set_env(env)
        model.learn(
            total_timesteps=end - model.num_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )
        done = model.num_timesteps
        env.close()

    if model is None:
        raise RuntimeError("nothing to train: total_timesteps already reached")
    model.save(run_dir / "final")
    callback.run_evaluation(final=True)
    return model


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m training.train",
        description="Train PPO from a committed experiment config.",
    )
    parser.add_argument("--config", default=None, help="experiment JSON file")
    parser.add_argument(
        "--resume", default=None, help="run directory to continue from latest.zip"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.resume:
        run_dir = Path(args.resume)
        cfg = load_experiment_config(run_dir / "config.json")
        if args.config:
            given = load_experiment_config(args.config)
            if given.to_dict() != cfg.to_dict():
                raise SystemExit("--config differs from the run's config.json")
    elif args.config:
        cfg = load_experiment_config(args.config)
        run_dir = run_dir_for(cfg)
    else:
        raise SystemExit("give --config (new run) or --resume DIR")
    train(cfg, run_dir, resume=bool(args.resume))
    print(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
