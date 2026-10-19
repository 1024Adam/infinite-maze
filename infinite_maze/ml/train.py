"""Training script for the Infinite Maze RL agent.

Usage
-----
# Phase 0 — short smoke run
python -m infinite_maze.ml.train --timesteps 1000

# Phase 1 — fresh training
python -m infinite_maze.ml.train --timesteps 100000

# Phase 2+ — resume from checkpoint
python -m infinite_maze.ml.train --timesteps 200000 --resume checkpoints/ppo_maze_100000_steps.zip

# Override phase (affects reward shaping in the environment)
python -m infinite_maze.ml.train --timesteps 750000 --phase 3 --n-envs 4

All hyperparameter defaults match the training plan phase table.
All constants sourced from config.ML_CONFIG — nothing hardcoded here.
"""

import argparse
import os
from datetime import datetime

# Suppress pygame hello banner before any other import
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from ..utils.config import config
from .environment import InfiniteMazeEnv

_ML = config.ML_CONFIG

CHECKPOINT_DIR = "checkpoints"


def _run_dir(phase: int) -> str:
    """Return a unique per-run subdirectory path, e.g. checkpoints/phase3_20260330_143022."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(CHECKPOINT_DIR, f"phase{phase}_{stamp}")


def _make_env(phase: int):
    def _factory():
        return InfiniteMazeEnv(phase=phase)

    return _factory


def _build_model(env, args) -> PPO:
    """Construct a fresh PPO model with CLI-supplied or default hyperparameters."""
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        device=args.device,
        tensorboard_log=_ML["TENSORBOARD_LOG"] if args.tensorboard else None,
        verbose=1,
    )


def _build_callbacks(eval_env, run_dir: str, args):
    callbacks = []

    os.makedirs(run_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=run_dir,
        name_prefix="ppo_maze",
        verbose=1,
    )
    callbacks.append(checkpoint_cb)

    if args.eval_freq > 0:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best"),
            log_path=os.path.join(run_dir, "eval_logs"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=_ML["EVAL_EPISODES"],
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_cb)

    return callbacks


def train(args) -> PPO:
    run_dir = _run_dir(args.phase)

    # Build vectorised training environment
    train_env = DummyVecEnv([_make_env(args.phase) for _ in range(args.n_envs)])

    # Separate single-env for evaluation (not vectorised). Wrap with Monitor so
    # EvalCallback reports canonical episode reward/length metrics.
    eval_env = Monitor(InfiniteMazeEnv(phase=args.phase))

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # Load model without env to skip observation-space check (old model has 53 features)
        model = PPO.load(args.resume, device=args.device, verbose=1)
        # Update model's spaces to match new env (backward-compat for feature upgrades)
        model.observation_space = train_env.observation_space
        model.action_space = train_env.action_space
        # Set env after updating spaces; policy will adapt to the new 56-feature space
        model.set_env(train_env)
        # Restore tensorboard log dir if configured
        if args.tensorboard:
            model.tensorboard_log = _ML["TENSORBOARD_LOG"]
    else:
        model = _build_model(train_env, args)

    callbacks = _build_callbacks(eval_env, run_dir, args)

    print(
        f"Training for {args.timesteps} timesteps (phase={args.phase}, n_envs={args.n_envs})"
    )
    print(f"Run directory: {run_dir}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=args.resume is None,
    )

    # Save final model
    os.makedirs(run_dir, exist_ok=True)
    final_path = os.path.join(run_dir, f"ppo_maze_final_{args.timesteps}_steps")
    model.save(final_path)
    print(f"Final model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()
    return model


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a PPO agent to play Infinite Maze.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    p.add_argument(
        "--timesteps",
        type=int,
        default=_ML["DEFAULT_TIMESTEPS"],
        help="Total environment steps to train for.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .zip checkpoint to resume training from.",
    )
    p.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4, 5],
        help="Training phase (controls reward shaping in the environment).",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (use 4 for Phase 3).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for PPO (cpu, cuda, or auto). Default cpu is faster for MlpPolicy.",
    )

    # PPO hyperparameters (plan defaults; override per-phase as needed)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.035)
    p.add_argument("--clip-range", type=float, default=0.10)

    # Callbacks
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=_ML["CHECKPOINT_FREQ"],
        help="Save a checkpoint every N environment steps.",
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=_ML["EVAL_FREQ"],
        help="Run evaluation every N environment steps. Set 0 to disable.",
    )
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help=f"Enable TensorBoard logging to {_ML['TENSORBOARD_LOG']}.",
    )

    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
