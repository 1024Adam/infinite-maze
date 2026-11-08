"""
Game rules, independent of rendering, input devices and wall-clock time.

``Simulation.step()`` is the single implementation of movement, collision,
scrolling, pace and death. Human play, ``controlled_run``, replays and tests all
drive it, so they cannot drift apart.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..entities.maze import Line
from ..entities.player import Player
from ..utils.config import config
from ..utils.seed import newSeed
from .game import Game

Intent = Tuple[int, int]  # (dx, dy), each in {-1, 0, 1}

RIGHT = config.get_movement_constant("RIGHT")
LEFT = config.get_movement_constant("LEFT")
UP = config.get_movement_constant("UP")
DOWN = config.get_movement_constant("DOWN")


@dataclass
class StepResult:
    """Outcome of one ``Simulation.step()``."""

    score_delta: int  # actual change to the game score (floor at 0 preserved)
    moved: Tuple[float, float]  # actual player displacement from input, pre-scroll
    terminated: bool  # the player was pushed past X_MIN
    truncated: bool  # max_steps was reached


class Simulation:
    """One game episode: the maze, player and rules, advanced by ``step()``."""

    def __init__(
        self,
        seed: Optional[int] = None,
        start_pace: int = config.START_PACE,
        max_steps: Optional[int] = config.MAX_EPISODE_STEPS,
        game: Optional[Game] = None,
        player: Optional[Player] = None,
    ) -> None:
        """Create a simulation, using a headless Game/Player unless injected."""
        self.game = game if game is not None else Game(headless=True)
        self.player = (
            player
            if player is not None
            else Player(config.PLAYER_START_X, config.PLAYER_START_Y, headless=True)
        )
        self.startPace = start_pace
        self.maxSteps = max_steps

        self.explicitSeed = seed
        self.seed = 0
        self.ticks = 0
        self.steps = 0
        self.terminated = False
        self.scrollOffset = 0
        self.lines: List[Line] = []
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """Start a new episode.

        An explicit seed (given here or at construction) is reused by every
        later reset; without one each episode draws a fresh random seed.
        """
        if seed is not None:
            self.explicitSeed = seed
        self.seed = newSeed() if self.explicitSeed is None else self.explicitSeed

        self.game.reset()
        self.game.setPace(self.startPace)
        self.game.setSeed(self.seed)
        # Only an explicitly requested seed is worth showing to the player.
        self.game.setSeedVisible(self.explicitSeed is not None)
        self.player.reset(config.PLAYER_START_X, config.PLAYER_START_Y)
        self.ticks = 0
        self.steps = 0
        self.terminated = False
        self.scrollOffset = 0
        self.lines = Line.generateMaze(
            self.game,
            config.MAZE_ROWS,
            config.MAZE_COLS,
            random.Random(self.seed),
        )

    def step(self, intent: Intent) -> StepResult:
        """Advance one tick: pace, movement with collision, scroll, bounds, recycle."""
        game = self.game
        player = self.player
        dx, dy = intent

        self.ticks += 1
        if self.ticks % config.PACE_UPDATE_TICKS == 0:
            game.setPace(game.getPace() + 1)

        scoreBefore = game.getScore()
        startX, startY = player.getPosition()

        if dx > 0:
            if not self._isBlocked(RIGHT):
                player.moveX(player.getSpeed())
                game.incrementScore()
        elif dx < 0:
            if not self._isBlocked(LEFT):
                player.moveX(-player.getSpeed())
                game.decrementScore()
        if dy > 0:
            if not self._isBlocked(DOWN):
                player.moveY(player.getSpeed())
        elif dy < 0:
            if not self._isBlocked(UP):
                player.moveY(-player.getSpeed())

        moved = (player.getX() - startX, player.getY() - startY)

        if self.ticks % config.SCROLL_INTERVAL_TICKS == 0:
            player.setX(player.getX() - game.getPace())
            self._scrollLines(game.getPace())

        # Position adjustments (to prevent screen overflow)
        terminated = player.getX() < game.X_MIN
        if terminated:
            self.terminated = True
            game.end()
        if player.getX() > game.X_MAX:
            player.setX(game.X_MAX)
            self._scrollLines(player.getSpeed())
        player.setY(max(player.getY(), game.Y_MIN))
        player.setY(min(player.getY(), game.Y_MAX))

        self._recycleLines()

        self.steps += 1
        truncated = self.maxSteps is not None and self.steps >= self.maxSteps
        return StepResult(
            score_delta=game.getScore() - scoreBefore,
            moved=moved,
            terminated=terminated,
            truncated=truncated,
        )

    def getSeed(self) -> int:
        return self.seed

    def getTicks(self) -> int:
        return self.ticks

    def getSteps(self) -> int:
        return self.steps

    def getLines(self) -> List[Line]:
        return self.lines

    def getScrollOffset(self) -> int:
        """Total pixels the lines have scrolled left since ``reset()``.

        ``worldX = screenX + getScrollOffset()`` is constant for a wall as the
        world scrolls, and the player's world x is ``player.getX() + offset``.
        Recycling moves a passed wall one maze tile (a period of
        ``2 * max(MAZE_ROWS, 3) * MAZE_CELL_SIZE`` pixels) to the right without
        changing the offset, so a wall's world x is well defined modulo that period.
        """
        return self.scrollOffset

    def isTerminated(self) -> bool:
        return self.terminated

    def getClosestLineX(self) -> int:
        """X of the nearest wall start ahead of the player, or a sentinel if none."""
        closest = config.NO_LINE_SENTINEL_X
        for line in self.lines:
            lineX = line.getXStart()
            if self.player.getX() < lineX and lineX < closest:
                closest = lineX
        return closest

    def getState(self) -> Dict[str, Any]:
        """Compact, JSON-friendly snapshot used by replays."""
        return {
            "score": self.game.getScore(),
            "ticks": self.ticks,
            "pace": self.game.getPace(),
            "player": [_plainNumber(v) for v in self.player.getPosition()],
            "terminated": self.terminated,
        }

    def _scrollLines(self, amount: int) -> None:
        """Shift every wall left by ``amount`` pixels."""
        self.scrollOffset += amount
        for line in self.lines:
            line.setXStart(line.getXStart() - amount)
            line.setXEnd(line.getXEnd() - amount)

    def _recycleLines(self) -> None:
        """Move lines that have been passed to the far right edge of the maze."""
        threshold = config.PLAYER_START_X
        passed = [line for line in self.lines if line.start[0] < threshold]
        if not passed:
            return
        # The target is the far edge before any line moves, as before.
        xMax = Line.getXMax(self.lines)
        for line in passed:
            start = line.getXStart()
            end = line.getXEnd()
            line.setXStart(xMax)
            if start == end:
                line.setXEnd(xMax)
            else:
                line.setXEnd(xMax + config.MAZE_CELL_SIZE)

    def _isBlocked(self, direction: int) -> bool:
        """Whether the player is blocked from moving one step in ``direction``.

        Comparison semantics (including the exact ``==`` checks on the leading
        edge) are inherited from the original engine and covered by golden traces.
        """
        player = self.player
        x = player.getX()
        y = player.getY()
        width = player.getWidth()
        height = player.getHeight()
        speed = player.getSpeed()

        for line in self.lines:
            xStart = line.getXStart()
            yStart = line.getYStart()
            xEnd = line.getXEnd()
            yEnd = line.getYEnd()
            horizontal = line.getIsHorizontal()

            overlapsY = (yStart <= y <= yEnd) or (yStart <= y + height <= yEnd)
            overlapsX = (xStart <= x <= xEnd) or (xStart <= x + width <= xEnd)

            if direction == RIGHT:
                if horizontal:
                    hit = y <= yStart <= y + height and x + width + speed == xStart
                else:
                    hit = (x + width <= xStart <= x + width + speed) and overlapsY
            elif direction == LEFT:
                if horizontal:
                    hit = y <= yStart <= y + height and x - speed == xEnd
                else:
                    hit = (x >= xEnd >= x - speed) and overlapsY
            elif direction == DOWN:
                if horizontal:
                    hit = (y + height <= yStart <= y + height + speed) and overlapsX
                else:
                    hit = x <= xStart <= x + width and y + height + speed == yStart
            else:  # UP
                if horizontal:
                    hit = (y >= yStart >= y - speed) and overlapsX
                else:
                    hit = x <= xStart <= x + width and y - speed == yEnd
            if hit:
                return True
        return False


def _plainNumber(value: float) -> Any:
    """Return integral floats as ints so state snapshots serialize cleanly."""
    return int(value) if float(value).is_integer() else value
