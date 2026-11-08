"""Wall-clock timing for the HUD timer and FPS limiting.

Game rules (pace, scrolling) do not use this clock; they are tick based and live
in ``Simulation``.
"""

import pygame


class Clock:
    """Wraps ``pygame.time.Clock`` and tracks elapsed milliseconds and frames."""

    def __init__(self) -> None:
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()

        self.prevMillis = 0
        self.millis = 0
        self.millisPaused = 0
        self.ticks = 0

    def update(self) -> None:
        """Advance one frame: accumulate elapsed milliseconds and count a tick."""
        self.prevMillis = self.millis
        self.time.tick()
        self.millis += self.time.get_time()
        self.tick()

    def getTimeString(self) -> str:
        """Return the elapsed time formatted as ``MM:SS``."""
        minutes = int(self.millis / 60000)
        seconds = int((self.millis / 1000) % 60)
        return f"{minutes:02}" + ":" + f"{seconds:02}"

    def getFps(self) -> float:
        return self.time.get_fps()

    def reset(self) -> None:
        """Restart the clock: zero all elapsed time and the tick counter."""
        self.time = pygame.time.Clock()
        self.startTime = self.time.get_time()

        self.prevMillis = 0
        self.millis = 0
        self.millisPaused = 0
        self.ticks = 0

    def getPrevMillis(self) -> int:
        return self.prevMillis

    def getMillis(self) -> int:
        return self.millis

    def getSeconds(self) -> int:
        return int((self.millis / 1000) % 60)

    def getPrevSeconds(self) -> int:
        return int((self.prevMillis / 1000) % 60)

    def rollbackMillis(self, rollback: int) -> None:
        """Remove ``rollback`` ms from the total, e.g. to exclude paused time."""
        self.millis -= rollback

    def getMillisPaused(self) -> int:
        return self.millisPaused

    def setMillisPaused(self, millis: int) -> None:
        self.millisPaused = millis

    def tick(self) -> None:
        """Count one frame."""
        self.ticks += 1

    def getTicks(self) -> int:
        return self.ticks
