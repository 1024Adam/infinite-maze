# Infinite Maze Game

[![Tests](https://github.com/1024Adam/infinite-maze/actions/workflows/tests.yml/badge.svg)](https://github.com/1024Adam/infinite-maze/actions/workflows/tests.yml)
[![Code Quality](https://github.com/1024Adam/infinite-maze/actions/workflows/quality.yml/badge.svg)](https://github.com/1024Adam/infinite-maze/actions/workflows/quality.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A fast-paced, infinitely challenging maze navigation game built with Python and Pygame.

## 🎮 Overview
Infinite Maze is a dynamic maze-style game where players navigate through a randomly generated maze, attempting to progress as far as possible while maintaining pace. The maze is infinite - as long as you can keep up with the advancing pace, the game continues indefinitely.

**What makes this truly challenging:** The game's pace increases periodically (every 1800 game ticks), forcing players to move faster while navigating increasingly complex maze layouts. Can you keep up with the relentless advance?

## ✨ Key Features
- **Infinite Gameplay**: Procedurally generated mazes that never end
- **Progressive Difficulty**: Pace increases periodically for escalating challenge
- **Smart Scoring**: Gain points for rightward progress, lose points for backtracking
- **Responsive Controls**: Smooth WASD or arrow key movement
- **Pause System**: Take a breather when needed
- **Modern Architecture**: Clean, extensible Python codebase with Poetry dependency management

## 🚀 Quick Start

### One-Command Setup (Recommended)
Get up and running in seconds:

```bash
# Linux/macOS
./setup.sh

# Windows
setup.bat
```

These scripts will automatically:
- Check if Poetry is installed (and guide you to install it if needed)
- Install all project dependencies
- Verify everything works correctly
- Launch the game

### Manual Installation
If you prefer more control over the installation process:

```bash
# 1. Install Poetry (one-time setup per machine)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Install dependencies and run the game
poetry install
poetry run infinite-maze
```

### Alternative Running Methods
Once installed, you can run the game multiple ways:

```bash
# Modern entry point (recommended)
poetry run infinite-maze

# Python module execution
python -m infinite_maze

# Direct execution (legacy, still supported)
python run_game.py
```

### Command-Line Options
```bash
infinite-maze [--seed N] [--start-pace N] [--record PATH] [--replay PATH] [--headless]
```

| Option | Description |
|---|---|
| `--seed N` | Generate the same maze every time (also on restart). The seed is shown at the bottom-left of the screen; without `--seed` every episode gets a new random seed, which is only logged (not shown) |
| `--start-pace N` | Start at pace `N` instead of 0 (debugging) |
| `--record PATH` | Write a replay file for each episode: `PATH`, `PATH-2`, `PATH-3`, ... |
| `--replay PATH` | Play a recorded replay back in a window |
| `--headless` | With `--replay`: no window; prints the final state and exits non-zero if it differs from the replay's `expected` state |

A replay is a small JSON file, so a bug can be reported as one file and reproduced with `infinite-maze --replay bug.json --headless`. Local recordings under `replays/` are git-ignored.

### Development Setup
```bash
# Install with development dependencies
poetry install

# Install test tooling into Poetry env
poetry run pip install -r requirements-test.txt

# Code formatting
poetry run black infinite_maze/

# Code linting
poetry run flake8 infinite_maze/

# Type checking
poetry run mypy infinite_maze/

# CI-parity quality gate
poetry run black --check infinite_maze/
poetry run flake8 infinite_maze/
poetry run mypy infinite_maze/
```

### Testing
```bash
# Install test dependencies (inside Poetry env)
poetry run pip install -r requirements-test.txt

# Run quick tests (unit + integration)
poetry run python run_tests.py

# Run all tests including performance benchmarks
poetry run python run_tests.py all

# Run with coverage report
poetry run python run_tests.py unit --coverage

# Run specific test categories
poetry run pytest -m unit                    # Unit tests only
poetry run pytest -m integration             # Integration tests only
poetry run pytest -m functional              # End-to-end tests
poetry run pytest -m performance --run-performance  # Performance benchmarks

# Match CI test selection (excludes slow/performance)
poetry run pytest -m "not slow and not performance"
```

The test suite includes:
- **293 test methods** across 4 categories
- **Unit tests**: Individual component testing
- **Integration tests**: Component interaction testing  
- **Functional tests**: End-to-end workflow testing
- **Performance tests**: Efficiency and benchmark testing

### First-Time Developer Setup
If you're new to the project:

1. **Clone the repository**
2. **Run the setup script**: `./setup.sh` (Linux/macOS) or `setup.bat` (Windows)
3. **Start developing!**

The setup script will:
- Check if Poetry is installed (and guide you to install it if needed)
- Install all project dependencies
- Verify everything works correctly

## 🎯 Game Mechanics

### Core Elements
|  Element   |  Description  |
| ---------- | ------------- |
| **Player** | Your character (red dot) that moves through the maze |
| **Pace Line** | The advancing boundary that pursues you - stay ahead or the game ends |
| **Maze Walls** | Dynamically generated barriers that block your movement |
| **Scoring** | Points awarded for rightward progress, deducted for leftward movement |

### Game Rules
- **Objective**: Navigate as far right through the infinite maze as possible
- **Scoring System**: 
  - +1 point for each rightward movement
  - -1 point for each leftward movement (minimum score: 0)
- **Pace Mechanic**: 
  - Increases every 1800 game ticks (frames); paused time does not count
  - Forces continuous forward progress
- **Game Over**: When the pace line catches up to your position

### Strategy Tips
- **Plan Ahead**: Look for openings in the maze before the pace forces movement
- **Minimize Backtracking**: Leftward movement costs points and wastes time
- **Use Vertical Space**: Moving up/down doesn't affect score but can find better paths
- **Stay Calm**: The pace increases, but panicking leads to poor decisions

## ⌨️ Controls
|  Key(s)        |  Action                 |
| -------------- | ----------------------- |
| **W, A, S, D** | Move up, left, down, right |
| **Arrow Keys** | Alternative movement controls |
| **Space**      | Pause/unpause game |
| **Esc, Q**     | Quit game |
| **Y, N**       | Yes/No responses when prompted |

## 📋 System Requirements
- **Python**: 3.8+ (tested with Python 3.8-3.12)
- **Dependencies**: Pygame 2.5+ (automatically managed by Poetry)
- **Platforms**: Windows, macOS, Linux
- **Memory**: ~50MB RAM
- **Storage**: ~10MB disk space

## 📚 Documentation

### Quick Links
- **[📖 Complete Documentation](docs/)** - Comprehensive guides for players and developers
- **[🎮 Player Guide](docs/player-guide.md)** - Master the game with strategies and tips
- **[⚙️ Installation Guide](docs/installation-guide.md)** - Detailed setup instructions
- **[🛠️ Developer Guide](docs/development.md)** - Development environment and workflow
- **[🔧 Troubleshooting](docs/troubleshooting.md)** - Solve common issues

### For Different Audiences
- **New Players**: Start with [Installation Guide](docs/installation-guide.md), then [Player Guide](docs/player-guide.md)
- **Developers**: Check [Development Guide](docs/development.md) and [Architecture Guide](docs/architecture.md)
- **Contributors**: Read [Contributing Guide](docs/contributing.md) and [API Reference](docs/api-reference.md)

## Development

### Code Quality Tools
- **Black**: Code formatting
- **Flake8**: Code linting and style checking
- **MyPy**: Static type checking
- **Pre-commit**: Git hooks for code quality (optional)

### Task Runners (Optional)
For convenience, you can use modern task runners instead of remembering Poetry commands:

#### Just (Recommended)
```bash
# Install Just: cargo install just
just setup     # Setup project
just run       # Run the game
just format    # Format code
just check     # Run all quality checks
```

#### Make (Traditional)
If you prefer the traditional approach, basic commands:
```bash
# You can create your own Makefile with Poetry commands
poetry install  # instead of 'make install'
poetry run infinite-maze  # instead of 'make run'
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper type hints and documentation
4. Run the code quality tools
5. Submit a pull request


## Dependencies
- **Pygame**: Game development library for graphics, sound, and input handling

For more information about Pygame, visit [pygame.org](https://www.pygame.org/docs/).

---

**Infinite Maze Game** - Evolution, not revolution. Modern Python practices with classic gameplay.
