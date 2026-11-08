"""The player entity: position, size, speed and sprite."""

import pygame
from typing import Optional, Tuple

from ..utils.config import config


class Player:
    """The player-controlled square; movement rules live in ``Simulation``."""

    def __init__(
        self, xPosition: float, yPosition: float, headless: bool = False
    ) -> None:
        """Place the player; ``headless`` skips loading the sprite."""
        self.headless = headless
        self.cursor: Optional[pygame.Surface]
        if not headless:
            try:
                image_path = config.get_image_path("player")
                self.cursor = pygame.image.load(image_path)
            except (pygame.error, FileNotFoundError):
                # If player image can't be loaded, create a simple shape
                self.cursor = pygame.Surface(
                    (config.PLAYER_WIDTH, config.PLAYER_HEIGHT)
                )
                self.cursor.fill(pygame.Color(255, 0, 0))  # Red square
        else:
            self.cursor = None

        self.position = (xPosition, yPosition)
        self.width = config.PLAYER_WIDTH
        self.height = config.PLAYER_HEIGHT
        self.speed = config.PLAYER_SPEED

    def setX(self, xPosition: float) -> None:
        self.position = (xPosition, self.position[1])

    def setY(self, yPosition: float) -> None:
        self.position = (self.position[0], yPosition)

    def moveX(self, units: float) -> None:
        """Move horizontally by ``units`` steps (scaled by the player's speed)."""
        new_x = self.position[0] + (units * self.speed)
        self.position = (new_x, self.position[1])

    def moveY(self, units: float) -> None:
        """Move vertically by ``units`` steps (scaled by the player's speed)."""
        new_y = self.position[1] + (units * self.speed)
        self.position = (self.position[0], new_y)

    def getX(self) -> float:
        return self.position[0]

    def getY(self) -> float:
        return self.position[1]

    def getPosition(self) -> Tuple[float, float]:
        return self.position

    def getSpeed(self) -> int:
        return self.speed

    def setCursor(self, image: str) -> None:
        """Load a new sprite from ``image``; keep the current one if it fails."""
        if not self.headless:
            try:
                self.cursor = pygame.image.load(image)
            except (pygame.error, FileNotFoundError):
                # If image can't be loaded, keep current cursor
                pass

    def getCursor(self) -> Optional[pygame.Surface]:
        return self.cursor

    def getWidth(self) -> int:
        return self.width

    def getHeight(self) -> int:
        return self.height

    def reset(self, xPosition: float, yPosition: float) -> None:
        """Move the player back to a position and restore the default speed."""
        self.setX(xPosition)
        self.setY(yPosition)
        self.speed = config.PLAYER_SPEED
