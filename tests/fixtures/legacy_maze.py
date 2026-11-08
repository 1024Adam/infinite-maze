"""Frozen reference implementation of ``Line.generateMaze``.

This is the original (slow) algorithm, copied verbatim. The optimised
``Line.generateMaze`` must produce identical lines (coordinates and side labels)
and draw random numbers in exactly the same sequence. Do not "improve" this file;
delete it only if you deliberately accept changing the maze of every seed.
"""

import random
from typing import Any, List, Optional

from infinite_maze.entities.maze import Line
from infinite_maze.utils.config import config


def legacyGenerateMaze(
    game: Any,
    width: int,
    height: int,
    rng: Optional[random.Random] = None,
) -> List[Line]:
    """Original ``Line.generateMaze`` (rescans all lines after every removal)."""
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
            (config.MAZE_CELL_SIZE * x) + config.PLAYER_START_X + config.MAZE_CELL_SIZE
        )
        for y in range(1, height - 1):
            yPos = (config.MAZE_CELL_SIZE * y) + game.Y_MIN
            lines.append(
                Line((xPos, yPos), (xPos + config.MAZE_CELL_SIZE, yPos), sideA, sideB)
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
                Line((xPos, yPos), (xPos, yPos + config.MAZE_CELL_SIZE), sideA, sideB)
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
