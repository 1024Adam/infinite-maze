"""Game state: display, score, pace, seed and pause/over flags, plus HUD rendering."""

import pygame
from typing import List, Optional

from .clock import Clock
from ..utils.config import config
from ..entities.player import Player
from ..entities.maze import Line


class Game:
    """Holds game-wide state and draws the screen; the rules live in Simulation."""

    # Display config
    WIDTH = config.SCREEN_WIDTH
    HEIGHT = config.SCREEN_HEIGHT
    X_MIN = config.PLAYER_START_X
    Y_MIN = config.MAZE_TOP_Y

    X_MAX = WIDTH / 2
    Y_MAX = HEIGHT - config.ICON_SIZE

    SCORE_INCREMENT = 1

    BG_COLOR = pygame.Color(255, 255, 255)
    FG_COLOR = pygame.Color(0, 0, 0)

    def __init__(self, headless: bool = False) -> None:
        """Create the game; ``headless`` skips the window, font and icon."""
        self.font: Optional[pygame.font.Font]
        self.screen: Optional[pygame.Surface]
        self.icon: Optional[pygame.Surface]
        if not headless:
            pygame.init()

            # Font Config
            self.font = pygame.font.SysFont("", config.PLAYER_HEIGHT * 2)

            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

            self.icon = pygame.Surface((config.ICON_SIZE, config.ICON_SIZE))
            try:
                iconImage = pygame.image.load(config.get_image_path("icon"))
                try:
                    self.icon.blit(iconImage, (0, 0))
                except TypeError:
                    # Some tests provide lightweight mock surfaces that pygame
                    # cannot blit; keep the default icon surface in that case.
                    pass
            except (pygame.error, FileNotFoundError):
                # If icon can't be loaded, use a default surface
                self.icon.fill(pygame.Color(255, 255, 255))

            pygame.display.set_caption("Infinite Maze")
            pygame.display.set_icon(self.icon)
        else:
            # Headless mode - minimal pygame initialization
            pygame.init()
            self.font = None
            self.screen = None
            self.icon = None

        self.pace = 0
        self.score = 0
        self.seed = 0
        self.seedVisible = False

        self.paused = False
        self.over = False
        self.shutdown = False
        self.headless = headless

        self.clock = Clock()

    def _updateClock(self) -> None:
        # The clock only drives the HUD timer and FPS limiting; game rules
        # (pace, scrolling) are tick based and live in Simulation.
        self.clock.update()
        if self.paused:
            self.clock.rollbackMillis(
                self.clock.getMillis() - self.clock.getPrevMillis()
            )

    def limitFps(self) -> None:
        """Sleep for the remainder of the frame; never called by the rules."""
        processTime = int(self.clock.getMillis() - self.clock.getPrevMillis())
        if processTime <= config.FPS_DELAY_MS:
            pygame.time.delay(config.FPS_DELAY_MS - processTime)

    def updateScreen(self, player: Player, lines: List[Line]) -> None:
        """Render one frame (maze, player, borders, HUD) and update the clock."""
        if self.headless:
            # In headless mode, just update the clock
            self._updateClock()
            return

        if self.screen is None or self.font is None:
            return

        screen = self.screen

        # Paint Screen
        screen.fill(self.BG_COLOR)
        cursor = player.getCursor()
        if cursor is not None:
            screen.blit(cursor, player.getPosition())
        for line in lines:
            pygame.draw.line(screen, self.FG_COLOR, line.getStart(), line.getEnd(), 1)

        self._updateClock()

        # Print Border
        pygame.draw.line(
            screen,
            self.FG_COLOR,
            (self.X_MIN, self.Y_MIN),
            (self.WIDTH, self.Y_MIN),
            2,
        )
        pygame.draw.line(
            screen,
            self.FG_COLOR,
            (self.X_MIN, self.Y_MAX + config.BORDER_OFFSET),
            (self.WIDTH, self.Y_MAX + config.BORDER_OFFSET),
            2,
        )
        pygame.draw.line(
            screen,
            self.FG_COLOR,
            (config.PLAYER_START_X, self.Y_MIN),
            (config.PLAYER_START_X, self.Y_MAX + config.BORDER_OFFSET),
            2,
        )

        # Print Display Text
        timeText = self.font.render(
            "Time: " + self.clock.getTimeString(), 1, self.FG_COLOR
        )
        self.screen.blit(timeText, (config.TEXT_MARGIN, config.TEXT_MARGIN))
        score_text = f"Score: {self.score}"
        scoreText = self.font.render(score_text, 1, self.FG_COLOR)
        self.screen.blit(scoreText, (config.TEXT_MARGIN, config.TEXT_MARGIN + 15))

        if self.seedVisible:
            seedText = self.font.render(f"Seed: {self.seed}", 1, self.FG_COLOR)
            self.screen.blit(
                seedText,
                (
                    config.TEXT_MARGIN,
                    self.Y_MAX + config.BORDER_OFFSET + config.SEED_TEXT_GAP,
                ),
            )

        if self.paused:
            scoreText = self.font.render(
                "Paused (press space to continue)", 1, self.FG_COLOR
            )
            self.screen.blit(scoreText, (100, config.TEXT_MARGIN))

        pygame.display.flip()

    def printEndDisplay(self) -> None:
        """Show the "Continue? (y/n)" screen with the final score."""
        if self.screen is None or self.font is None:
            return

        # Paint Screen
        self.screen.fill(self.BG_COLOR)
        endText = self.font.render("Continue? (y/n)", 1, self.FG_COLOR)
        score_text = f"Score: {self.score}"
        scoreText = self.font.render(score_text, 1, self.FG_COLOR)

        # Print Display Text
        self.screen.blit(endText, (config.TEXT_MARGIN, config.TEXT_MARGIN))
        self.screen.blit(scoreText, (config.TEXT_MARGIN, config.TEXT_MARGIN + 15))

        pygame.display.flip()

    def end(self) -> None:
        """Mark the current episode as over."""
        self.over = True

    def cleanup(self) -> None:
        """Shut pygame down."""
        pygame.quit()

    def isActive(self) -> bool:
        return not self.over

    def quit(self) -> None:
        """Request that the application exit."""
        self.shutdown = True

    def isPlaying(self) -> bool:
        return not self.shutdown

    def reset(self) -> None:
        """Start a fresh episode: zero pace and score, clear over, reset the clock."""
        self.pace = 0
        self.score = 0
        self.over = False

        self.clock.reset()

    def getClock(self) -> Clock:
        return self.clock

    def getScreen(self) -> Optional[pygame.Surface]:
        return self.screen

    def getScore(self) -> int:
        return self.score

    def updateScore(self, amount: int) -> None:
        self.score += amount

    def incrementScore(self) -> None:
        self.score += self.SCORE_INCREMENT

    def decrementScore(self) -> None:
        """Lower the score by one; it never goes below zero."""
        self.score -= self.SCORE_INCREMENT if self.score > 0 else 0

    def setScore(self, newScore: int) -> None:
        self.score = newScore

    def isPaused(self) -> bool:
        return self.paused

    def changePaused(self, player: Player) -> None:
        """Toggle pause, swapping the HUD colour and the player sprite."""
        self.paused = not self.paused
        if self.paused:
            self.FG_COLOR = pygame.Color(128, 128, 128)
            player.setCursor(config.get_image_path("player_paused"))
        else:
            self.FG_COLOR = pygame.Color(0, 0, 0)
            player.setCursor(config.get_image_path("player"))

    def getPace(self) -> int:
        return self.pace

    def setPace(self, newPace: int) -> None:
        self.pace = newPace

    def getSeed(self) -> int:
        return self.seed

    def setSeed(self, newSeed: int) -> None:
        self.seed = newSeed

    def setSeedVisible(self, visible: bool) -> None:
        """Choose whether the HUD shows the seed."""
        self.seedVisible = visible

    def isSeedVisible(self) -> bool:
        return self.seedVisible
