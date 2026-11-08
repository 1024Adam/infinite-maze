"""
Launcher script for Infinite Maze game.

This script allows running the game with: python run_game.py
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    from infinite_maze.core.engine import maze
    maze()
