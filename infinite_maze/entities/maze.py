"""Maze entities: wall segments (``Line``) and seedable maze generation."""

import random
from typing import Any, List, Optional, Tuple

from ..utils.config import config


class Line:
    """A wall segment, with the two cell sets it separates during generation."""

    def __init__(
        self,
        startPos: Tuple[int, int] = (0, 0),
        endPos: Tuple[int, int] = (0, 0),
        sideA: int = 0,
        sideB: int = 0,
    ) -> None:
        self.start = startPos
        self.end = endPos
        self.sideA = sideA
        self.sideB = sideB
        self.isHorizontal = startPos[1] == endPos[1]

    def getStart(self) -> Tuple[int, int]:
        return self.start

    def setStart(self, newStart: Tuple[int, int]) -> None:
        self.start = newStart

    def getEnd(self) -> Tuple[int, int]:
        return self.end

    def setEnd(self, newEnd: Tuple[int, int]) -> None:
        self.end = newEnd

    def getXStart(self) -> int:
        return self.start[0]

    def setXStart(self, newX: int) -> None:
        self.start = (newX, self.start[1])

    def getYStart(self) -> int:
        return self.start[1]

    def setYStart(self, newY: int) -> None:
        self.start = (self.start[0], newY)

    def getXEnd(self) -> int:
        return self.end[0]

    def setXEnd(self, newX: int) -> None:
        self.end = (newX, self.end[1])

    def getYEnd(self) -> int:
        return self.end[1]

    def setYEnd(self, newY: int) -> None:
        self.end = (self.end[0], newY)

    def getSideA(self) -> int:
        return self.sideA

    def setSideA(self, side: int) -> None:
        self.sideA = side

    def getSideB(self) -> int:
        return self.sideB

    def setSideB(self, side: int) -> None:
        self.sideB = side

    def getIsHorizontal(self) -> bool:
        return self.isHorizontal

    def resetIsHorizontal(self) -> None:
        """Recompute the horizontal flag after the endpoints were changed."""
        self.isHorizontal = self.start[1] == self.end[1]

    @staticmethod
    def getXMax(lines: List["Line"]) -> int:
        """Return the largest end x among ``lines`` (0 for an empty list)."""
        if not lines:
            return 0
        xMax = lines[0].getXEnd()  # Initialize with first line's end
        for line in lines:
            lineEnd = line.getXEnd()
            if lineEnd > xMax:
                xMax = lineEnd
        return xMax

    @staticmethod
    def generateMaze(
        game: Any,
        width: int,
        height: int,
        rng: Optional[random.Random] = None,
    ) -> List["Line"]:
        """Generate maze walls by removing random walls between separate cell sets.

        The result is a spanning tree of cells. The same seeded ``rng`` always
        gives the same maze; ``rng=None`` uses a fresh unseeded generator.
        """
        if rng is None:
            rng = random.Random()
        lines: List[Line] = []
        # Keep generation stable for tiny test dimensions.
        width = max(width, 3)
        height = max(height, 3)
        # Cells per column; also the cell-id stride between columns.
        stride = height - 1

        # Horizontal Line Gen
        for x in range(width * 2):
            sideA = (stride * x) + 1
            sideB = sideA + 1

            xPos = (
                (config.MAZE_CELL_SIZE * x)
                + config.PLAYER_START_X
                + config.MAZE_CELL_SIZE
            )
            for y in range(1, height - 1):
                yPos = (config.MAZE_CELL_SIZE * y) + game.Y_MIN
                lines.append(
                    Line(
                        (xPos, yPos), (xPos + config.MAZE_CELL_SIZE, yPos), sideA, sideB
                    )
                )
                sideA = sideB
                sideB += 1
        # Vertical Line Gen
        for y in range(height - 1):
            sideA = y + 1
            sideB = sideA + stride

            yPos = (config.MAZE_CELL_SIZE * y) + game.Y_MIN
            for x in range(1, width * 2):
                xPos = (
                    (config.MAZE_CELL_SIZE * x)
                    + config.PLAYER_START_X
                    + config.MAZE_CELL_SIZE
                )
                lines.append(
                    Line(
                        (xPos, yPos), (xPos, yPos + config.MAZE_CELL_SIZE), sideA, sideB
                    )
                )
                sideA = sideB
                sideB += stride

        # Create 'maze' structure
        # (will be complete when all 'cells' are connected to each other)
        sets: List[int] = []
        if not lines:
            return lines

        while len(sets) != 1:
            length = len(lines)
            lineNum = rng.randint(0, length - 1)
            tempSideA = lines[lineNum].getSideA()
            tempSideB = lines[lineNum].getSideB()
            if tempSideA != tempSideB:
                del lines[lineNum]
                for line in lines:
                    if line.getSideA() == tempSideB:
                        line.setSideA(tempSideA)
                    if line.getSideB() == tempSideB:
                        line.setSideB(tempSideA)
            sets = []
            for line in lines:
                tempSideA = line.getSideA()
                tempSideB = line.getSideB()
                if tempSideA not in sets:
                    sets.append(tempSideA)
                if tempSideB not in sets:
                    sets.append(tempSideB)

        return lines
