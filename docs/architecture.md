# Infinite Maze - Technical Architecture

## Overview
Infinite Maze follows modern Python packaging standards with a clean, modular architecture designed for maintainability, extensibility, and performance. The game is built using the Pygame library with a clear separation of concerns across different layers.

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Game Engine Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │ Input Handling  │  │ Game Loop       │  │ Rendering      │
│  │ (pygame events) │  │ (engine.py)     │  │ (pygame)       │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Game Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │ Game State      │  │ Clock/Timing    │  │ Collision      │
│  │ (game.py)       │  │ (clock.py)      │  │ Detection      │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Entity Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │ Player Entity   │  │ Maze Generation │  │ Line Entities  │
│  │ (player.py)     │  │ (maze.py)       │  │ (maze.py)      │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Utilities Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │ Configuration   │  │ Logging         │  │ Asset Loading  │
│  │ (config.py)     │  │ (logger.py)     │  │ (config.py)    │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

### Package Organization
```
infinite_maze/
├── infinite_maze/              # Main game package
│   ├── __init__.py             # Package exports and version info
│   ├── __main__.py             # Entry point for python -m infinite_maze
│   │
│   ├── core/                   # Core game logic
│   │   ├── __init__.py
│   │   ├── engine.py           # Human / controlled_run adapters (input, rendering, FPS)
│   │   ├── simulation.py       # Game rules: Simulation.step() (no clock, no display)
│   │   ├── replay.py           # Record / replay of games
│   │   ├── game.py             # Game state management
│   │   └── clock.py            # HUD timer and frame rate control
│   │
│   ├── entities/               # Game entities
│   │   ├── __init__.py
│   │   ├── player.py           # Player character logic
│   │   ├── maze.py             # Seedable maze generation and line entities
│   │   └── maze_analysis.py    # Geometry-based maze verification (connectivity, solvability)
│   │
│   └── utils/                  # Utility modules
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       └── logger.py           # Centralized logging
│
├── assets/                     # Game assets
│   └── images/                 # Sprite and icon files
│       ├── icon.png
│       ├── player.png
│       └── player_paused.png
│
├── docs/                       # Project documentation
├── pyproject.toml              # Modern Python project configuration
├── poetry.lock                 # Dependency lock file
├── run_game.py                 # Legacy entry point (compatibility)
└── README.md                   # Main project documentation
```

## 🎮 Core Components

### 1. Game Engine (`core/engine.py`)
**Purpose**: Adapters that connect input and output to the rules in `Simulation`

**Key Responsibilities**:
- Main game loop execution
- Keyboard input processing, converted to an `Intent` `(dx, dy)`
- Game state transitions (playing, paused, game over)
- Frame rate management (`time.delay`) - never inside the rules
- Optional recording of the intents that reach `Simulation.step()`

**Key Functions**:
- `maze()`: Main entry point and primary game loop
- `controlled_run()`: Alternative loop for external control (AI/testing)

### 1b. Simulation (`core/simulation.py`)
**Purpose**: The single implementation of the game rules.

`Simulation.step(intent)` performs, in order: tick and pace update, horizontal then vertical movement with collision, scroll every `SCROLL_INTERVAL_TICKS` ticks, death check (`x < X_MIN`), `X_MAX` clamp, `Y` clamp, recycling of passed lines, and the optional `max_steps` truncation check. It returns a `StepResult`.

- **Seeded**: a game is reproducible from `(seed, intents)`. An explicit seed regenerates the same maze on restart; no seed means a new random seed each episode.
- **Performance**: `Line.generateMaze` tracks cell-set labels incrementally (about 5 ms per maze instead of about 780 ms) while keeping the exact random draw sequence and output of the original algorithm; `tests/unit/test_maze_equivalence.py` compares it with the frozen reference in `tests/fixtures/legacy_maze.py`. `_recycleLines` only scans for the far edge when a line actually needs recycling. `Simulation.getScrollOffset()` exposes the accumulated scroll so external code (the oracle in `training/`) can convert screen x to world x.
- **Tick-based time**: pace advances every `PACE_UPDATE_TICKS` (`PACE_UPDATE_INTERVAL * FPS` = 1800) ticks; this is a frame count, not a wall-clock duration, so the real time between increases depends on the achieved frame rate, scrolling happens when `ticks % SCROLL_INTERVAL_TICKS == 0`. There is no sleep and no wall clock, so simulations run as fast as the CPU allows. Paused frames do not call `step()` and therefore do not advance ticks.
- **Shared by**: human play (`maze()`), `controlled_run()`, replays and tests.

```
 keys / wrapper.control / replay file
            │  Intent (dx, dy)
            ▼
   Simulation.step(intent) ──► StepResult(score_delta, moved, terminated, truncated)
            │
   Line.generateMaze(..., rng=random.Random(seed))
```

### 1c. Replays (`core/replay.py`)
A versioned JSON file `{version, seed, start_pace, maze, intents (run-length encoded [dx, dy, count]), expected}`. `ReplayRecorder` collects intents, `loadReplay` validates version and maze configuration (a mismatch is an error, not silent divergence), `runReplay` plays it back with or without a window. See `infinite-maze --record/--replay`.

### 2. Game State Manager (`core/game.py`)
**Purpose**: Central game state and configuration management

**Key Responsibilities**:
- Game state tracking (active, paused, game over)
- Score management and calculation
- Holds pace, score and seed values (pace progression itself is in `Simulation`)
- Screen rendering coordination (HUD shows time and score, plus the seed when one was passed explicitly)
- Display configuration

**Key Features**:
- Headless mode support for testing/AI
- Pause state management with visual feedback
- Progressive difficulty through pace acceleration
- Screen boundary enforcement

### 3. Player Entity (`entities/player.py`)
**Purpose**: Player character representation and movement

**Key Responsibilities**:
- Player position tracking
- Movement calculation and validation
- Sprite management and rendering
- Player state persistence

**Key Features**:
- Smooth movement with configurable speed
- Visual state changes (normal/paused sprites)
- Collision boundary calculation
- Position reset for game restart

### 4. Maze System (`entities/maze.py`)
**Purpose**: Procedural maze generation and wall management

**Key Responsibilities**:
- Random maze layout generation
- Wall positioning and repositioning
- Collision boundary definition
- Infinite maze illusion through line recycling

**Key Features**:
- Dynamic line repositioning for infinite effect
- Configurable maze density and complexity
- Efficient wall collision detection
- Maze boundary management

### 5. Timing System (`core/clock.py`)
**Purpose**: Game timing, frame rate control, and pace management

**Key Responsibilities**:
- Frame rate regulation
- Game time tracking
- Pause state time handling (HUD timer only; pace is tick based, see `Simulation`)

### 6. Configuration System (`utils/config.py`)
**Purpose**: Centralized configuration management

**Key Features**:
- Color palette management
- Asset path resolution
- Game mechanics constants
- Control mapping definitions
- Extensible configuration architecture

## 🔄 Data Flow

### 1. Game Initialization
```
Engine Start → Game Object Creation → Player Spawn → Maze Generation → Clock Reset
```

### 2. Main Game Loop
```
Input Processing → Intent → Simulation.step (pace/tick update → movement with
collision → scroll → bounds → recycle) → Screen Rendering → Frame Rate Control
```

### 3. Maze Management
```
Line Position Check → Out-of-bounds Detection → Line Repositioning → 
Collision Boundary Update → Render Update
```

### 4. Scoring System
```
Movement Input → Direction Analysis → Score Calculation → 
Display Update → State Persistence
```

## 🎯 Design Patterns

### 1. **Entity-Component Pattern**
- Player, Game, and Maze components are separate, focused entities
- Each component handles its own state and behavior
- Clean interfaces between components

### 2. **Configuration Pattern**
- Centralized configuration eliminates magic numbers
- Easy customization without code changes
- Type-safe configuration access

### 3. **State Machine Pattern** (Game States)
- Clear state transitions: Active → Paused → Game Over → Reset
- State-specific behavior isolation
- Predictable state management

### 4. **Factory Pattern** (Maze Generation)
- `Line.generateMaze()` creates maze layouts
- Consistent maze structure generation
- Easy to extend with different maze algorithms

## ⚡ Performance Considerations

### 1. **Collision Detection Optimization**
- Early termination when collision detected
- Boundary-based collision rather than pixel-perfect
- Minimal collision checks per frame

### 2. **Rendering Efficiency**
- Only render visible elements
- Efficient pygame surface management
- Minimal screen updates

### 3. **Memory Management**
- Object reuse for maze lines
- Minimal object creation in main loop
- Efficient asset loading

### 4. **Frame Rate Control**
- Target 60 FPS with dynamic delay adjustment
- Smooth animation through consistent timing
- CPU usage optimization

## 🧪 Testing Architecture

### Test Suite Overview
The project includes a comprehensive test suite with **293 test methods** across four distinct categories:

### Test Categories
1. **Unit Tests (184 tests)**: Individual component testing in isolation
   - Player movement and collision bounds
   - Game state management and scoring
   - Clock timing and FPS monitoring
   - Maze generation algorithms
   - Configuration management

2. **Integration Tests (69 tests)**: Component interaction testing
   - Player-maze collision detection
   - Game engine coordination
   - Asset loading integration
   - State synchronization between components

3. **Functional Tests (21 tests)**: End-to-end workflow testing
   - Complete gameplay scenarios
   - Game state transitions (pause/resume, game over)
   - User interaction simulation

4. **Performance Tests (19 tests)**: Efficiency and benchmark validation
   - Frame rate stability (60 FPS target)
   - Memory usage monitoring (<200MB target)
   - Collision detection speed (>1M checks/sec)
   - Game loop efficiency (<16.67ms/frame)

### Test Infrastructure
- **Pygame Mocking**: Headless testing without actual display/audio
- **Performance Monitoring**: Built-in benchmarking utilities
- **Deterministic Testing**: Controlled timing and input simulation
- **CI/CD Ready**: Automated testing environment support

### Test Structure
```
tests/
├── conftest.py                    # Pytest configuration
├── fixtures/                     # Test utilities
│   ├── game_fixtures.py          # Game-specific fixtures
│   ├── pygame_mocks.py           # Pygame mocking framework
│   └── test_helpers.py           # Common utilities
├── unit/                         # Component isolation tests
├── integration/                  # Component interaction tests
├── functional/                   # End-to-end workflow tests
└── performance/                  # Efficiency benchmarks
```

### Testable Components
- **Configuration System**: Asset paths, game settings validation
- **Maze Generation**: Algorithm correctness and performance
- **Collision Detection**: Accuracy and efficiency testing
- **Scoring System**: Point calculation and boundary enforcement
- **Game State Management**: Pause/resume, game over, restart workflows
- **Player Movement**: Position tracking, boundary validation
- **Clock System**: Timing accuracy, FPS monitoring, rollback functionality

### Running Tests
```bash
# Quick development tests
python run_tests.py

# Comprehensive testing
python run_tests.py all --coverage

# Category-specific testing
python run_tests.py unit
python run_tests.py integration
python run_tests.py functional
python run_tests.py performance
```

## 🔧 Extension Points

### 1. **New Game Modes**
- Implement new game classes inheriting from base `Game`
- Add mode-specific logic while reusing core components

### 2. **Alternative Maze Algorithms**
- Extend `Line` class with new generation methods
- Plugin architecture for maze generators

### 3. **Enhanced Graphics**
- Replace pygame surfaces with sprites
- Add particle effects and animations
- Implement themes and visual customization

### 4. **AI Integration**
- Use `controlled_run()` function for AI training
- Implement observation and action interfaces
- Add learning environment wrappers

## 📊 Dependencies

### Runtime Dependencies
- **Pygame 2.5+**: Core game engine and rendering
- **Python 3.8+**: Language runtime

### Development Dependencies
- **Poetry**: Dependency management and packaging
- **Black**: Code formatting
- **Flake8**: Code linting and style checking
- **MyPy**: Static type checking

## 🚀 Future Architecture Considerations

### Planned Enhancements
1. **Plugin System**: Extensible game mechanics
2. **Network Multiplayer**: Client-server architecture
3. **Advanced Graphics**: 3D rendering support
4. **Mobile Support**: Cross-platform deployment
5. **Save System**: Game progress persistence

### Scalability Considerations
- Modular component architecture supports feature additions
- Configuration system enables easy customization
- Clean interfaces allow component replacement
- Modern Python packaging supports distribution

---

This architecture provides a solid foundation for both gameplay and development, balancing simplicity with extensibility while maintaining high performance and code quality.
