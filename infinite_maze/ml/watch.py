"""Watch a trained PPO agent play Infinite Maze with real-time rendering.

Usage
-----
python -m infinite_maze.ml.watch --model checkpoints/ppo_maze_final_10000_steps.zip

# Watch multiple episodes
python -m infinite_maze.ml.watch --model checkpoints/ppo_maze_final_10000_steps.zip --episodes 5

# Slow it down (default 16 ms ≈ 60 fps; increase for slower playback)
python -m infinite_maze.ml.watch --model checkpoints/ppo_maze_final_10000_steps.zip --delay 50

# Press ESC or close the window at any time to stop.
"""

import argparse

import pygame
from stable_baselines3 import PPO

from ..core.game import Game
from ..entities.player import Player
from ..entities.maze import Line
from ..utils.config import config
from .environment import InfiniteMazeEnv
from .features import (
    get_obs,
    is_blocked_right,
    is_blocked_left,
    is_blocked_up,
    is_blocked_down,
)

_ML = config.ML_CONFIG
_MC = config.MOVEMENT_CONSTANTS
RIGHT = _MC["RIGHT"]
LEFT = _MC["LEFT"]
UP = _MC["UP"]
DOWN = _MC["DOWN"]
DO_NOTHING = _MC["DO_NOTHING"]


def _user_quit() -> bool:
    """Return True if the user closed the window or pressed ESC."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def watch(
    model_path: str,
    n_episodes: int,
    phase: int,
    delay_ms: int,
    device: str,
    legacy_obs: bool = False,
) -> None:
    """Run n_episodes of a trained model in a live pygame window.

    The game loop mirrors core/engine.py exactly:
      1. updateScreen() — renders the frame, advances clock, updates real-time pace
      2. Collision check — replicated from environment.py / features.py
      3. Action application — moveX/moveY, score update
      4. Pace shift — every PACE_SHIFT_INTERVAL ticks via the real clock
      5. Position adjustments — X_MAX clamp, Y clamp
      6. Line recycling
      7. Termination check

    Pace builds on the real-time clock (30 s per increment), matching the
    live game experience rather than the accelerated tick-based training pace.

    Parameters
    ----------
    legacy_obs : bool
        When True, pass ``legacy=True`` to ``get_obs()`` so the observation
        shape is (54,) instead of (57,). Use this to watch models trained
        before the history features (indices 13-15) and gap offset (index 16) were added.
    """
    if legacy_obs:
        # Load without env-validation — the network expects (53,) inputs,
        # which is incompatible with the current InfiniteMazeEnv (56,).
        model = PPO.load(model_path, device=device)
    else:
        dummy = InfiniteMazeEnv(phase=phase)
        model = PPO.load(model_path, env=dummy, device=device)
        dummy.close()

    # Real game objects — display window opened once, reused across episodes
    game = Game(headless=False)
    player = Player(config.PLAYER_START_X, config.PLAYER_START_Y, headless=False)

    game.getClock().reset()

    for ep in range(n_episodes):
        game.reset()
        player.reset(config.PLAYER_START_X, config.PLAYER_START_Y)
        lines = Line.generateMaze(game, config.MAZE_ROWS, config.MAZE_COLS)
        consecutive_blocked = 0

        print(f"\nEpisode {ep + 1} / {n_episodes}  (ESC or close window to stop)")

        while game.isActive():
            if _user_quit():
                game.cleanup()
                return

            # -- Render (also advances clock and updates real-time pace) --
            game.updateScreen(player, lines)

            # -- Build observation --
            obs = get_obs(player, lines, game, consecutive_blocked, legacy=legacy_obs)

            # -- Model prediction --
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # -- Collision flags before the move --
            blocked_right = is_blocked_right(player, lines)
            blocked_left = is_blocked_left(player, lines)
            blocked_up = is_blocked_up(player, lines)
            blocked_down = is_blocked_down(player, lines)

            # -- Apply action --
            if action == RIGHT and not blocked_right:
                player.moveX(1)
                game.incrementScore()
                consecutive_blocked = 0
            elif action == LEFT and not blocked_left:
                player.moveX(-1)
                game.decrementScore()
            elif action == UP and not blocked_up:
                player.moveY(-1)
            elif action == DOWN and not blocked_down:
                player.moveY(1)

            if action == RIGHT and blocked_right:
                consecutive_blocked += 1

            # -- Pace shift (mirrors engine.py: every 10 clock ticks) --
            ticks = game.getClock().getTicks()
            if ticks % _ML["PACE_SHIFT_INTERVAL"] == 0:
                shift = game.getPace()
                player.setX(player.getX() - shift)
                for line in lines:
                    line.setXStart(line.getXStart() - shift)
                    line.setXEnd(line.getXEnd() - shift)

            # -- Position adjustments --
            if player.getX() > int(game.X_MAX):
                player.setX(int(game.X_MAX))
                for line in lines:
                    line.setXStart(line.getXStart() - player.getSpeed())
                    line.setXEnd(line.getXEnd() - player.getSpeed())

            player.setY(max(player.getY(), game.Y_MIN))
            player.setY(min(player.getY(), game.Y_MAX))

            # -- Line recycling --
            x_max = Line.getXMax(lines)
            for line in lines:
                start = line.getXStart()
                end = line.getXEnd()
                if start < config.PLAYER_START_X:
                    line.setXStart(x_max)
                    if start == end:
                        line.setXEnd(x_max)
                    else:
                        line.setXEnd(x_max + config.MAZE_CELL_SIZE)

            # -- Termination --
            if player.getX() < game.X_MIN:
                game.end()

            # -- Frame pacing --
            if delay_ms > 0:
                pygame.time.delay(delay_ms)

        print(f"  Score: {game.getScore()}  |  Pace reached: {game.getPace()}")

    game.cleanup()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Watch a trained PPO agent play Infinite Maze.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", type=str, required=True, help="Path to a saved .zip model file."
    )
    p.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to watch."
    )
    p.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4, 5],
        help="Environment phase used when loading the model.",
    )
    p.add_argument(
        "--delay",
        type=int,
        default=16,
        help="Milliseconds to wait between frames. 16≈60fps, 50≈20fps.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for PPO load/predict (cpu, cuda, or auto).",
    )
    p.add_argument(
        "--legacy-obs",
        action="store_true",
        default=False,
        help="Use 53-feature observation (no history features). Required for "
        "models trained before the action-repeat/x-progress/blocked-right "
        "history features were added.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    watch(
        args.model, args.episodes, args.phase, args.delay, args.device, args.legacy_obs
    )


if __name__ == "__main__":
    main()
