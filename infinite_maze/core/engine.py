"""
Adapters between the outside world and ``Simulation``.

``maze()`` (human play) and ``controlled_run()`` (external controller) own input,
rendering, pausing and FPS limiting. All game rules live in ``Simulation.step()``.
"""

import pygame
from typing import Any, Dict, Optional

from ..entities.player import Player
from ..utils import logger
from ..utils.config import config
from .game import Game
from .replay import ReplayRecorder, episodePath, saveReplay
from .simulation import Intent, Simulation

DO_NOTHING = config.get_movement_constant("DO_NOTHING")
RIGHT = config.get_movement_constant("RIGHT")
LEFT = config.get_movement_constant("LEFT")
UP = config.get_movement_constant("UP")
DOWN = config.get_movement_constant("DOWN")


def _intentFromKeys(keys: Any) -> Intent:
    """Horizontal and vertical keys are independent, so diagonals are possible."""
    dx = 0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        dx = 1
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
        dx = -1
    dy = 0
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        dy = 1
    elif keys[pygame.K_UP] or keys[pygame.K_w]:
        dy = -1
    return (dx, dy)


def _actionFromKeys(keys: Any) -> Optional[int]:
    """Map held keys to a single discrete action, or None if no move key is held."""
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return RIGHT
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return LEFT
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return DOWN
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return UP
    return None


def _startRecorder(
    sim: Simulation, recordPath: Optional[str]
) -> Optional[ReplayRecorder]:
    """Log the episode seed and return a recorder if recording was requested."""
    logger.info(f"Episode seed: {sim.getSeed()}")
    if recordPath is None:
        return None
    return ReplayRecorder(sim.getSeed(), sim.startPace)


def _saveRecording(
    recorder: ReplayRecorder, sim: Simulation, recordPath: str, episode: int
) -> None:
    """Write the finished episode's replay, numbering files after the first."""
    path = episodePath(recordPath, episode)
    saveReplay(recorder.toReplay(sim.getState()), path)
    logger.info(f"Replay saved to {path}")


def maze(
    seed: Optional[int] = None,
    start_pace: int = config.START_PACE,
    record_path: Optional[str] = None,
) -> None:
    """Run the interactive game: keyboard input, rendering, pause and restart."""
    # Contains All Game Stats/Config
    game = Game()

    # Player Stats/Position/Details
    player = Player(config.PLAYER_START_X, config.PLAYER_START_Y)

    # Rules, maze and seed. Human play is never truncated.
    sim = Simulation(
        seed=seed, start_pace=start_pace, max_steps=None, game=game, player=player
    )
    episode = 1
    recorder = _startRecorder(sim, record_path)

    keys = pygame.key.get_pressed()
    while game.isPlaying():
        while game.isActive():
            game.updateScreen(player, sim.getLines())
            prevKeys = keys
            keys = pygame.key.get_pressed()

            if not game.isPaused():
                intent = _intentFromKeys(keys)
                result = sim.step(intent)
                if recorder is not None:
                    recorder.record(intent)
                if result.truncated:
                    game.end()

            # Pause Event
            if prevKeys[pygame.K_SPACE] and not keys[pygame.K_SPACE]:
                game.changePaused(player)

            # Quit Events
            if keys[pygame.K_LMETA] and keys[pygame.K_q]:
                game.end()
            if keys[pygame.K_RMETA] and keys[pygame.K_q]:
                game.end()
            if keys[pygame.K_LALT] and keys[pygame.K_F4]:
                game.end()
            if keys[pygame.K_ESCAPE]:
                game.end()

            # Process Game Events
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                game.end()

            game.limitFps()

        # Game has ended
        if recorder is not None and record_path is not None:
            _saveRecording(recorder, sim, record_path, episode)
            recorder = None
        game.printEndDisplay()

        # Quit Events
        keys = pygame.key.get_pressed()
        if keys[pygame.K_y]:
            sim.reset()
            episode += 1
            recorder = _startRecorder(sim, record_path)
        if keys[pygame.K_n]:
            game.quit()
        if keys[pygame.K_LMETA] and keys[pygame.K_q]:
            game.quit()
        if keys[pygame.K_RMETA] and keys[pygame.K_q]:
            game.quit()
        if keys[pygame.K_LALT] and keys[pygame.K_F4]:
            game.quit()

        # Process Game Events
        if any(event.type == pygame.QUIT for event in pygame.event.get()):
            game.quit()

    game.cleanup()
    exit(0)


def controlled_run(wrapper: Any, counter: int, seed: Optional[int] = None) -> None:
    """Run one game where ``wrapper.control(values)`` chooses each action.

    Held keyboard keys override the wrapper's action; ``wrapper.gameover`` is
    called with the final score.
    """
    # Contains All Game Stats/Config
    game = Game()

    # Player Stats/Position/Details
    player = Player(config.PLAYER_START_X, config.PLAYER_START_Y)

    sim = Simulation(seed=seed, game=game, player=player)
    logger.info(f"Episode seed: {sim.getSeed()}")

    keys = pygame.key.get_pressed()
    action: Optional[int] = None
    while game.isPlaying():
        while game.isActive():
            game.updateScreen(player, sim.getLines())
            keys = pygame.key.get_pressed()

            values: Dict[str, Any] = {}
            if not game.isPaused():
                # Keyboard input overrides the controller's action
                keyAction = _actionFromKeys(keys)
                if keyAction is not None:
                    action = keyAction
                elif action is None:
                    action = DO_NOTHING

                values["action"] = action
                result = sim.step(config.ACTION_TO_INTENT.get(action, (0, 0)))
                values["score_increased"] = result.score_delta > 0
                if result.truncated:
                    game.end()

            game.limitFps()

            values["closest_line"] = sim.getClosestLineX()

            response = wrapper.control(values)
            action = response

        game.quit()

    wrapper.gameover(game.getScore())
