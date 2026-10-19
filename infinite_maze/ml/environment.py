"""Gymnasium environment wrapper for the Infinite Maze game.

Headless-safe: never calls pygame.display.set_mode(), pygame.display.flip(),
or any rendering method. Drives the game step-by-step through direct state
manipulation — the live engine loop (core/engine.py) is never imported or used.

Observation space: Box(57,) float32, all values in [0.0, 1.0].
Action space:      Discrete(5) mapping to config.MOVEMENT_CONSTANTS.
"""

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..core.game import Game
from ..entities.player import Player
from ..entities.maze import Line
from ..utils.config import config
from .features import (
    get_obs,
    is_blocked_right,
    is_blocked_left,
    is_blocked_up,
    is_blocked_down,
    nearest_right_gap_offset,
    bfs_optimal_action,
)
from .rewards import compute_reward, phase3_shaping, phase4_direction_shaping

_ML = config.ML_CONFIG
_MC = config.MOVEMENT_CONSTANTS

DO_NOTHING = _MC["DO_NOTHING"]
RIGHT = _MC["RIGHT"]
LEFT = _MC["LEFT"]
UP = _MC["UP"]
DOWN = _MC["DOWN"]


class InfiniteMazeEnv(gym.Env):
    """Gymnasium environment wrapping the Infinite Maze game.

    Parameters
    ----------
    phase : int
        Training phase (1–5). Phase 3 activates path-finding reward shaping.
        All other phases use the base reward function from rewards.py.
    render_mode : None
        Rendering is not supported; value must be None.
    """

    metadata = {"render_modes": []}

    def __init__(self, phase: int = 1, render_mode=None):
        super().__init__()

        self.phase = phase

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(57,), dtype=np.float32
        )

        # Game objects — initialised on first reset()
        self._game = None
        self._player = None
        self._lines = None

        # Episode state
        self._tick_counter = 0
        self._consecutive_blocked = 0
        self._consecutive_vertical = 0
        self._consecutive_vertical_any = 0
        self._consecutive_up = 0
        self._consecutive_down = 0
        self._last_vertical_action = -1
        self._corner_escape_dir = -1
        self._corner_escape_progress = 0
        self._prev_gap_offset = 0.5
        self._blocked_vertical_same_y_streak = 0
        self._blocked_vertical_anchor_y = -1
        self._blocked_do_nothing_streak = 0
        self._recent_actions = deque(maxlen=int(_ML["ACTION_REPEAT_WINDOW"]))
        self._recent_x_positions = deque(maxlen=int(_ML["X_PROGRESS_WINDOW"]))
        self._recent_blocked_right = deque(maxlen=int(_ML["BLOCKED_RIGHT_WINDOW"]))

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Start a new episode.

        Parameters
        ----------
        seed : int | None
            RNG seed forwarded to gymnasium's base reset.
        options : dict | None
            Optional overrides:
            - ``"start_pace"`` (int): initial game pace. Used by Phase 4
              curriculum domain randomisation.
        """
        super().reset(seed=seed)

        self._game = Game(headless=True)
        self._player = Player(
            config.PLAYER_START_X, config.PLAYER_START_Y, headless=True
        )
        self._lines = Line.generateMaze(self._game, config.MAZE_ROWS, config.MAZE_COLS)

        self._tick_counter = 0
        self._consecutive_blocked = 0
        self._consecutive_vertical = 0
        self._consecutive_vertical_any = 0
        self._consecutive_up = 0
        self._consecutive_down = 0
        self._last_vertical_action = -1
        self._corner_escape_dir = -1
        self._corner_escape_progress = 0
        self._prev_gap_offset = 0.5
        self._blocked_vertical_same_y_streak = 0
        self._blocked_vertical_anchor_y = -1
        self._blocked_do_nothing_streak = 0
        self._recent_actions.clear()
        self._recent_x_positions.clear()
        self._recent_blocked_right.clear()
        self._recent_x_positions.append(self._player.getX())

        if options is not None and "start_pace" in options:
            self._game.setPace(int(options["start_pace"]))

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        """Advance the environment by one tick.

        Parameters
        ----------
        action : int
            Integer in [0, 4] mapping to config.MOVEMENT_CONSTANTS.

        Returns
        -------
        obs : np.ndarray  shape (57,) float32
        reward : float
        terminated : bool  True when player is pushed past X_MIN by pace
        truncated : bool   True when score >= EPISODE_SCORE_CAP
        info : dict        {'score', 'pace', 'tick'}
        """
        game = self._game
        player = self._player
        lines = self._lines
        prev_x = player.getX()
        prev_y = player.getY()

        # -- 1. Collision flags (computed before the move) --
        blocked_right = is_blocked_right(player, lines)
        blocked_left = is_blocked_left(player, lines)
        blocked_up = is_blocked_up(player, lines)
        blocked_down = is_blocked_down(player, lines)
        at_top_boundary = player.getY() <= game.Y_MIN
        at_bottom_boundary = player.getY() >= game.Y_MAX
        effective_blocked_up = blocked_up or at_top_boundary
        effective_blocked_down = blocked_down or at_bottom_boundary
        blocked_flags = {
            "right": blocked_right,
            "left": blocked_left,
            "up": effective_blocked_up,
            "down": effective_blocked_down,
        }

        if self.phase == 2:
            in_corner = blocked_right and (
                (effective_blocked_up and not effective_blocked_down)
                or (effective_blocked_down and not effective_blocked_up)
            )
            blocked_trigger = int(_ML["CORNER_ESCAPE_TRIGGER_BLOCKED"])
            if in_corner and self._consecutive_blocked >= blocked_trigger:
                if self._corner_escape_dir == -1:
                    self._corner_escape_dir = DOWN if effective_blocked_up else UP
                    self._corner_escape_progress = 0

        # -- 1b. BFS optimal action (Phase 3+ curriculum) --
        bfs_action = bfs_optimal_action(player, lines, game) if self.phase >= 3 else -1

        # -- 2. Apply action --
        if action == RIGHT and not blocked_right:
            player.moveX(1)
            game.incrementScore()
        elif action == LEFT and not blocked_left:
            player.moveX(-1)
            game.decrementScore()
        elif action == UP and not effective_blocked_up:
            player.moveY(-1)
        elif action == DOWN and not effective_blocked_down:
            player.moveY(1)
        # DO_NOTHING: no move

        # -- 3. Tick counter --
        self._tick_counter += 1

        # -- 4. Pace shift (every PACE_SHIFT_INTERVAL ticks) --
        if self._tick_counter % _ML["PACE_SHIFT_INTERVAL"] == 0:
            shift = game.getPace()
            player.setX(player.getX() - shift)
            for line in lines:
                line.setXStart(line.getXStart() - shift)
                line.setXEnd(line.getXEnd() - shift)

        # -- 5. Position adjustments (every step, mirrors engine.py) --
        if player.getX() > int(game.X_MAX):
            player.setX(int(game.X_MAX))
            for line in lines:
                line.setXStart(line.getXStart() - player.getSpeed())
                line.setXEnd(line.getXEnd() - player.getSpeed())

        player.setY(max(player.getY(), game.Y_MIN))
        player.setY(min(player.getY(), game.Y_MAX))

        # -- 6. Line recycling (every step) --
        x_max = Line.getXMax(lines)
        for line in lines:
            start = line.getXStart()
            end = line.getXEnd()
            if start < config.PLAYER_START_X:
                line.setXStart(x_max)
                if start == end:  # vertical line — xStart == xEnd
                    line.setXEnd(x_max)
                else:  # horizontal line — span one cell
                    line.setXEnd(x_max + config.MAZE_CELL_SIZE)

        # -- 7. Pace update (tick-based, replaces real-time Clock) --
        if self._tick_counter % _ML["TICKS_PER_PACE_UPDATE"] == 0:
            game.setPace(game.getPace() + 1)

        # -- 8 & 9. Terminal / truncated --
        terminated = player.getX() < game.X_MIN
        truncated = game.getScore() >= _ML["EPISODE_SCORE_CAP"]

        # -- 10. Consecutive-blocked counter update --
        if not blocked_right and action == RIGHT:
            new_blocked = 0
        elif blocked_right:
            new_blocked = self._consecutive_blocked + 1
        else:
            new_blocked = self._consecutive_blocked

        # -- 10b. Consecutive-vertical counter update --
        if action in (UP, DOWN):
            if action == self._last_vertical_action:
                new_vertical = self._consecutive_vertical + 1
            else:
                new_vertical = 1
            new_vertical_any = self._consecutive_vertical_any + 1
            new_up = (self._consecutive_up + 1) if action == UP else 0
            new_down = (self._consecutive_down + 1) if action == DOWN else 0
        else:
            new_vertical = 0
            new_vertical_any = 0
            new_up = 0
            new_down = 0

        # -- 10c. Phase 3+ blocked-right vertical stalling detector --
        if self.phase >= 3 and blocked_right and action in (UP, DOWN):
            y_delta = int(_ML["PHASE3_STUCK_VERTICAL_Y_DELTA"])
            new_y = player.getY()
            if self._blocked_vertical_anchor_y < 0:
                self._blocked_vertical_anchor_y = prev_y
            if abs(new_y - self._blocked_vertical_anchor_y) <= y_delta:
                self._blocked_vertical_same_y_streak += 1
            else:
                self._blocked_vertical_anchor_y = new_y
                self._blocked_vertical_same_y_streak = 1
        else:
            self._blocked_vertical_same_y_streak = 0
            self._blocked_vertical_anchor_y = -1

        # -- 11. Reward --
        prev_gap_offset = self._prev_gap_offset
        prev_blocked = self._consecutive_blocked

        reward = float(
            compute_reward(
                action,
                blocked_flags,
                terminated,
                game,
                bfs_action=bfs_action,
                phase=self.phase,
                player=self._player,
                recent_actions=self._recent_actions,
                recent_x_positions=self._recent_x_positions,
                blocked_right=blocked_right,
            )
        )

        # Penalise repeating the same vertical direction (breaks UP/DOWN looping exploit)
        if new_vertical > _ML["CONSECUTIVE_VERTICAL_CAP"]:
            reward += float(_ML["REWARD_CONSECUTIVE_VERTICAL_PENALTY"])

        # Penalise any vertical loitering regardless of direction (breaks UP/DOWN oscillation)
        if new_vertical_any > _ML["CONSECUTIVE_VERTICAL_ANY_CAP"]:
            reward += float(_ML["REWARD_CONSECUTIVE_VERTICAL_ANY_PENALTY"])

        # Penalise excessively long UP-only runs that indicate vertical drift.
        if self.phase >= 4 and new_up > int(_ML["PHASE4_CONSECUTIVE_UP_CAP"]):
            reward += float(_ML["REWARD_PHASE4_CONSECUTIVE_UP_PENALTY"])
        elif self.phase >= 3 and new_up > int(_ML["PHASE3_CONSECUTIVE_UP_CAP"]):
            reward += float(_ML["REWARD_PHASE3_CONSECUTIVE_UP_PENALTY"])

        # Penalise excessively long DOWN-only runs that indicate vertical drift.
        if self.phase >= 3 and new_down > int(_ML["PHASE3_CONSECUTIVE_DOWN_CAP"]):
            reward += float(_ML["REWARD_PHASE3_CONSECUTIVE_DOWN_PENALTY"])

        # Penalise rapid vertical direction flips (UP -> DOWN or DOWN -> UP)
        # This encourages sustained directional commitment rather than alternation
        if action in (UP, DOWN) and self._last_vertical_action in (UP, DOWN):
            if action != self._last_vertical_action:
                reward += float(_ML["REWARD_VERTICAL_DIRECTION_FLIP"])

        # Penalise repeated blocked-right vertical attempts with little Y movement.
        if self.phase >= 3:
            stuck_window = int(_ML["PHASE3_STUCK_VERTICAL_WINDOW"])
            if self._blocked_vertical_same_y_streak > stuck_window:
                reward += float(_ML["REWARD_PHASE3_STUCK_VERTICAL"])

        # Phase 4: extra penalty when RIGHT remains blocked for too long.
        if self.phase >= 4 and blocked_right:
            blocked_cap = int(_ML["PHASE4_BLOCKED_RIGHT_CAP"])
            if new_blocked > blocked_cap:
                reward += float(_ML["REWARD_PHASE4_BLOCKED_RIGHT_STREAK"])

        # Penalise repeated DO_NOTHING when blocked-right (Phase 3+ idle exploit)
        if self.phase >= 3 and blocked_right:
            if action == DO_NOTHING:
                self._blocked_do_nothing_streak += 1
            else:
                self._blocked_do_nothing_streak = 0
            
            loop_window = int(_ML["PHASE3_DO_NOTHING_LOOP_WINDOW"])
            if self._blocked_do_nothing_streak > loop_window:
                reward += float(_ML["REWARD_PHASE3_DO_NOTHING_LOOP"])
        elif not blocked_right:
            self._blocked_do_nothing_streak = 0

        if self.phase == 2 and self._corner_escape_dir in (UP, DOWN):
            target = int(_ML["CORNER_ESCAPE_TARGET_STEPS"])
            if action == self._corner_escape_dir:
                # Attempting correct direction: small bonus, and progress if move succeeds
                reward += float(_ML["REWARD_CORNER_ESCAPE_ATTEMPT"])
                if action == UP and not effective_blocked_up:
                    self._corner_escape_progress += 1
                    reward += float(_ML["REWARD_CORNER_ESCAPE_STEP"])
                elif action == DOWN and not effective_blocked_down:
                    self._corner_escape_progress += 1
                    reward += float(_ML["REWARD_CORNER_ESCAPE_STEP"])
            elif action in (UP, DOWN):
                # Wrong direction: penalty (but lighter, and no progress regression)
                reward += float(_ML["REWARD_CORNER_ESCAPE_WRONG_DIR"])
            elif action in (LEFT, RIGHT):
                # Horizontal movement during escape: weak penalty
                reward += float(_ML["REWARD_CORNER_ESCAPE_HORIZONTAL"])
            elif action == DO_NOTHING:
                reward += float(_ML["REWARD_CORNER_ESCAPE_WAIT"])

            post_blocked_right = is_blocked_right(player, lines)
            if self._corner_escape_progress >= target and not post_blocked_right:
                reward += float(_ML["REWARD_CORNER_ESCAPE_CLEAR"])
                self._corner_escape_dir = -1
                self._corner_escape_progress = 0

        self._consecutive_blocked = new_blocked
        self._consecutive_vertical = new_vertical
        self._consecutive_vertical_any = new_vertical_any
        self._consecutive_up = new_up
        self._consecutive_down = new_down
        self._last_vertical_action = (
            action if action in (UP, DOWN) else self._last_vertical_action
        )
        self._recent_actions.append(action)
        self._recent_x_positions.append(player.getX())
        self._recent_blocked_right.append(float(blocked_right))

        # -- 12. Build observation and return --
        obs = self._get_obs()
        new_gap = nearest_right_gap_offset(player, lines, game)

        if self.phase >= 3:
            reward += phase3_shaping(
                action,
                prev_gap_offset,
                new_gap,
                prev_blocked,
                new_blocked,
                blocked_right,
                prev_x,
                player.getX(),
            )

        if self.phase >= 4:
            reward += phase4_direction_shaping(
                action,
                new_gap,
                blocked_right,
            )

        self._prev_gap_offset = new_gap

        info = {
            "score": game.getScore(),
            "pace": game.getPace(),
            "tick": self._tick_counter,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # headless — no rendering supported

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return get_obs(
            self._player,
            self._lines,
            self._game,
            self._consecutive_blocked,
            self._recent_actions,
            self._recent_x_positions,
            self._recent_blocked_right,
        )
