# Infinite Maze - Quick Reference

A handy reference card for players and developers.

## 🎮 Player Quick Reference

### ⚡ Getting Started (30 seconds)
1. **Install**: Run `setup.bat` (Windows) or `./setup.sh` (Linux/macOS)
2. **Play**: Game launches automatically, or run `poetry run infinite-maze`
3. **Move**: Use WASD or Arrow keys
4. **Survive**: Stay ahead of the advancing pace line!

### ⌨️ Controls
| Key(s) | Action | Score Impact |
|--------|--------|--------------|
| **W** or **↑** | Move Up | No change |
| **A** or **←** | Move Left | **-1 point** |
| **S** or **↓** | Move Down | No change |
| **D** or **→** | Move Right | **+1 point** |
| **Space** | Pause/Resume | - |
| **Esc** or **Q** | Quit Game | - |
| **Y/N** | Yes/No (game over) | - |

### 🎯 Game Rules
- **Goal**: Navigate as far right as possible through infinite maze
- **Pace**: Increases every 1800 game ticks (frames) of play
- **Scoring**: +1 for right, -1 for left, 0 for up/down
- **Game Over**: When pace line catches your position

### 🏆 Score Benchmarks
- **Beginner**: 0-50 points
- **Intermediate**: 50-150 points  
- **Advanced**: 150-300 points
- **Expert**: 300-500 points
- **Master**: 500+ points

### 💡 Quick Tips
- **Use pause** to study maze layout
- **Minimize backtracking** (left movement)
- **Plan vertical routes** to find better horizontal paths
- **Stay calm** as pace increases
- **Look ahead** for escape routes

---

## 🛠️ Developer Quick Reference

### ⚡ Development Setup (2 minutes)
```bash
# 1. Clone and setup
git clone https://github.com/1024Adam/infinite-maze.git
cd infinite-maze
poetry install

# 2. Run the game
poetry run infinite-maze

# 3. Code quality checks
poetry run black infinite_maze/
poetry run flake8 infinite_maze/
poetry run mypy infinite_maze/
```

### 🔧 Common Commands
```bash
# Running the game
poetry run infinite-maze        # Modern entry point
poetry run maze-game           # Alternative entry point
python -m infinite_maze        # Module execution
python run_game.py             # Legacy method

# Development tools
poetry install                 # Install dependencies
poetry update                  # Update dependencies
poetry run black infinite_maze/  # Format code
poetry run flake8 infinite_maze/ # Lint code
poetry run mypy infinite_maze/   # Type check

# Testing
pip install -r requirements-test.txt  # Install test deps (one-time)
python run_tests.py             # Quick tests (unit + integration)
python run_tests.py all         # All tests including performance
python run_tests.py unit --coverage  # Unit tests with coverage
pytest -m unit                  # Direct pytest unit tests

# Using Just task runner (optional)
just setup                     # Setup environment
just run                       # Run game
just check                     # All quality checks
just clean                     # Clean build artifacts
```

### 📁 Project Structure
```
infinite_maze/
├── infinite_maze/              # Main package
│   ├── core/                   # Game engine
│   │   ├── engine.py           # Main game loop
│   │   ├── game.py             # Game state
│   │   └── clock.py            # Timing system
│   ├── entities/               # Game objects
│   │   ├── player.py           # Player character
│   │   └── maze.py             # Maze generation
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration
│       └── logger.py           # Logging
├── assets/images/              # Game assets
├── docs/                       # Documentation
└── pyproject.toml              # Project config
```

### 🎯 Key Classes
```python
# Core classes
from infinite_maze import Game, Player, Line, Clock
from infinite_maze.utils.config import config
from infinite_maze.utils.logger import logger

# Basic usage
game = Game()                   # Create game instance
player = Player(80, 223)       # Create player
lines = Line.generateMaze(game, 15, 20)  # Generate maze
```

### 🔍 Configuration
```python
# In utils/config.py
class GameConfig:
    SCREEN_WIDTH = 800          # Window width
    SCREEN_HEIGHT = 600         # Window height
    PLAYER_SPEED = 5            # Movement speed
    FPS = 60                    # Target frame rate
    
    # Colors, controls, paths, etc.
```

---

## 🚨 Troubleshooting Quick Fixes

### Installation Issues
```bash
# Python not found
python --version                # Check version (need 3.8+)
which python                   # Check location

# Poetry not found  
curl -sSL https://install.python-poetry.org | python3 -
# Then restart terminal

# Pygame fails
pip install --upgrade pip      # Update pip first
pip install pygame             # Try direct install
```

### Runtime Issues
```bash
# Game won't start
poetry install                 # Reinstall dependencies
poetry run python -v -m infinite_maze  # Verbose output

# Performance problems
# Close other applications
# Update graphics drivers
# Try: poetry run infinite-maze --windowed (if supported)
```

### Development Issues
```bash
# Code quality fails
poetry run black infinite_maze/     # Fix formatting
poetry run flake8 infinite_maze/    # Check specific errors

# Git conflicts
git status                     # See conflicted files
git fetch upstream
git rebase upstream/main
```

---

## 📚 Documentation Quick Links

### For Players
- **[Complete Player Guide](docs/player-guide.md)** - Strategies, tips, and mastery
- **[Installation Guide](docs/installation-guide.md)** - Detailed setup instructions
- **[Troubleshooting](docs/troubleshooting.md)** - Solve common issues

### For Developers  
- **[Architecture Guide](docs/architecture.md)** - Technical design and patterns
- **[API Reference](docs/api-reference.md)** - Complete class documentation
- **[Contributing Guide](docs/contributing.md)** - How to contribute
- **[Development Guide](docs/development.md)** - Dev environment setup

### Project Info
- **[Main README](README.md)** - Project overview and quick start
- **[Documentation Index](docs/README.md)** - All documentation links

---

## 🎯 Entry Points Summary

### For Players
| Method | Command | Notes |
|--------|---------|-------|
| **Automatic** | `setup.bat` / `./setup.sh` | Easiest, runs setup + game |
| **Modern** | `poetry run infinite-maze` | Recommended after setup |
| **Module** | `python -m infinite_maze` | Alternative method |
| **Legacy** | `python run_game.py` | Backwards compatibility |

### For Developers
| Method | Command | Purpose |
|--------|---------|---------|
| **Development** | `poetry run infinite-maze` | Testing changes |
| **Debugging** | `poetry run python -v -m infinite_maze` | Verbose output |
| **Just Tasks** | `just run` | Task runner (if installed) |
| **Direct** | `python run_game.py` | Bypass Poetry |

---

## 💡 Pro Tips

### Gameplay
- Move right immediately - pace begins from the start
- Use vertical movement strategically
- Pause to plan complex routes
- Focus on survival over score initially

### Development
- Use Poetry for dependency management
- Run quality checks before committing
- Test changes manually before pushing
- Keep documentation updated

### Contributing
- Start with small changes
- Follow the contribution guidelines
- Test thoroughly
- Ask questions if unclear

---

*Keep this reference handy while playing or developing! For detailed information, see the complete documentation in the [docs/](docs/) directory.* 🎮
