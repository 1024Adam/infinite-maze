"""
Core game systems for Infinite Maze.

This package contains the fundamental game engine components:
- engine: Human and controlled game loops (input, rendering, FPS)
- simulation: Game rules (Simulation.step), independent of time and display
- replay: Record and replay of games
- game: Game state management
- clock: Timing and frame rate control
"""

from .engine import maze
from .game import Game
from .simulation import Simulation
from .clock import Clock

__all__ = ["maze", "Game", "Clock", "Simulation"]
