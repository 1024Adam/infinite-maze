# Infinite Maze - API Reference

This document provides comprehensive technical documentation for all classes, methods, and functions in the Infinite Maze codebase. This reference is intended for developers who want to understand, modify, or extend the game.

## 📋 Table of Contents

- [Core Module (`infinite_maze.core`)](#core-module-infinite_mazecore)
  - [Engine (`engine.py`)](#engine-enginepy)
  - [Simulation (`simulation.py`)](#simulation-simulationpy)
  - [Replay (`replay.py`)](#replay-replaypy)
  - [Game (`game.py`)](#game-gamepy)
  - [Clock (`clock.py`)](#clock-clockpy)
- [Entities Module (`infinite_maze.entities`)](#entities-module-infinite_mazeentities)
  - [Player (`player.py`)](#player-playerpy)
  - [Maze (`maze.py`)](#maze-mazepy)
- [Utils Module (`infinite_maze.utils`)](#utils-module-infinite_mazeutils)
  - [Config (`config.py`)](#config-configpy)
  - [Logger (`logger.py`)](#logger-loggerpy)
- [Constants & Enums](#constants--enums)

---

## Core Module (`infinite_maze.core`)

### Engine (`engine.py`)

The main game engine responsible for the game loop, input handling, and coordination between game components.

#### Constants

```python
DO_NOTHING = 0
RIGHT = 1
LEFT = 2
UP = 3
DOWN = 4
```

Movement direction constants used throughout the engine.

#### Functions

##### `maze(seed: Optional[int] = None, start_pace: int = 0, record_path: Optional[str] = None) -> None`

**Primary game entry point and main game loop.**

**Description:**
Initializes the game components and runs the main gameplay loop. Handles all user input, game state management, collision detection, and rendering.

**Parameters:**
- `seed` (int, optional): Seed for maze generation. An explicit seed regenerates the same maze on every restart; `None` draws a new random seed per episode. The seed is always logged, but only shown in the HUD when passed explicitly.
- `start_pace` (int): Initial pace.
- `record_path` (str, optional): Write a replay per episode (`path`, `path-2`, ...).

**Usage:**
```python
from infinite_maze.core.engine import maze
maze()  # Starts the game
maze(seed=1234)  # Same maze every time
```

**Game Loop Flow:**
1. Initialize game, player, and maze components
2. Process keyboard input (WASD/Arrow keys)
3. Convert keys to an `Intent` and call `Simulation.step()` (collision, scrolling, pace, bounds)
4. Render the game screen
6. Control frame rate
7. Handle game over and restart logic

**Key Features:**
- Real-time collision detection
- Progressive pace acceleration
- Pause/resume functionality
- Clean game state transitions

##### `controlled_run(wrapper, counter, seed: Optional[int] = None) -> None`

**Alternative game loop for external control (AI/testing).**

**Parameters:**
- `wrapper`: External controller object with `control()` and `gameover()` methods
- `counter`: Iteration counter for external tracking
- `seed`: Optional maze seed (same semantics as `maze(seed=...)`)

**Description:**
Provides a controlled version of the main game loop where movement decisions can be made by external systems (AI agents, automated testing, etc.). Returns game state information to the controller and processes the controller's movement decisions.

**Usage:**
```python
class AIController:
    def control(self, values):
        # Process game state and return movement decision
        return movement_action
    
    def gameover(self, final_score):
        # Handle game over event
        pass

ai = AIController()
controlled_run(ai, 0)
```

**State Information Provided:**
- Current action being performed
- Whether score was increased this frame
- Distance to closest wall/obstacle

---

### Simulation (`simulation.py`)

The single implementation of the game rules; see [architecture](architecture.md). No rendering, sleeping or wall-clock access.

```python
from infinite_maze.core.simulation import Simulation

sim = Simulation(seed=1234, start_pace=0, max_steps=None)
result = sim.step((1, 0))  # (dx, dy), each in {-1, 0, 1}
```

#### `StepResult`
| Field | Meaning |
|---|---|
| `score_delta` | Actual change of the score (the floor at 0 is honoured) |
| `moved` | Actual player displacement from input, before scrolling |
| `terminated` | The player was pushed past `X_MIN` |
| `truncated` | `max_steps` was reached |

#### `Simulation`
- `__init__(seed=None, start_pace=config.START_PACE, max_steps=config.MAX_EPISODE_STEPS, game=None, player=None)`
- `reset(seed=None)`: new episode. An explicit seed (here or in the constructor) is reused by later resets.
- `step(intent) -> StepResult`
- `getSeed()`, `getTicks()`, `getSteps()`, `getLines()`, `getClosestLineX()`, `isTerminated()`, `getState()`

Discrete actions map to intents through `config.ACTION_TO_INTENT`.

### Replay (`replay.py`)

```json
{
  "version": 1,
  "seed": 1234,
  "start_pace": 0,
  "maze": {"rows": 15, "cols": 20, "cell": 22},
  "intents": [[1, 0, 120], [0, 1, 7]],
  "expected": {"score": 85, "ticks": 127, "pace": 0, "player": [101, 223], "terminated": false}
}
```

`intents` are run-length encoded `[dx, dy, count]`; only steps that reach `Simulation.step()` are recorded (paused frames are not). `expected` is optional and makes the replay a regression test.

- `ReplayRecorder(seed, startPace)`: `record(intent)`, `toReplay(finalState=None)`
- `saveReplay(replay, path)`, `loadReplay(path)`: raises `ReplayError` on a bad file, unsupported version or maze-config mismatch
- `runReplay(replay, render=False) -> dict`: final state
- `compareExpected(replay, result) -> list[str]`: mismatches against `expected`
- `episodePath(basePath, episode)`

### Game (`game.py`)

Central game state management class handling all game configuration, scoring, timing, and rendering coordination.

#### Class: `Game`

##### Class Attributes

```python
# Display configuration
WIDTH = 640          # Screen width in pixels
HEIGHT = 480         # Screen height in pixels
X_MIN = 80          # Left boundary for player movement
Y_MIN = 40          # Top boundary for player movement
X_MAX = WIDTH / 2   # Right boundary for player movement
Y_MAX = HEIGHT - 32 # Bottom boundary for player movement

# Gameplay configuration
SCORE_INCREMENT = 1  # Points awarded per rightward movement

# Visual configuration
BG_COLOR = pygame.Color(255, 255, 255)  # Background color (white)
FG_COLOR = pygame.Color(0, 0, 0)        # Foreground color (black)
```

##### Constructor

```python
def __init__(self, headless=False)
```

**Parameters:**
- `headless` (bool): If True, runs without graphics for testing/AI (default: False)

**Description:**
Initializes the game state, pygame systems, fonts, display, and game variables.

**Example:**
```python
# Standard game initialization
game = Game()

# Headless mode for testing
test_game = Game(headless=True)
```

##### Methods

###### `updateScreen(player, lines) -> None`

**Updates and renders the game screen.**

**Parameters:**
- `player` (Player): Player object to render
- `lines` (List[Line]): List of maze line objects to render

**Description:**
Handles all screen rendering including player sprite, maze walls, UI text, borders, and timing updates. Also manages pace acceleration triggers.

**Features:**
- Automatic pace acceleration every `config.PACE_UPDATE_TICKS` game ticks
- Pause state visual feedback
- Score and time display
- Border and boundary rendering

###### `printEndDisplay() -> None`

**Renders the game over screen.**

**Description:**
Displays the final score and prompts user for restart decision.

###### State Management Methods

```python
def end() -> None                    # Trigger game over
def cleanup() -> None               # Clean up pygame resources
def isActive() -> bool              # Check if game is currently active
def quit() -> None                  # Quit the entire application
def isPlaying() -> bool             # Check if game should continue running
def reset() -> None                 # Reset game state for new game
```

###### Accessor Methods

```python
def getClock() -> Clock             # Get the game clock object
def getScreen() -> pygame.Surface   # Get the game screen surface
def getScore() -> int               # Get current score
def isPaused() -> bool              # Check if game is paused
def getPace() -> int                # Get current pace value
```

###### Score Management

```python
def updateScore(amount: int) -> None    # Add/subtract arbitrary amount
def incrementScore() -> None            # Add standard score increment
def decrementScore() -> None           # Subtract score (minimum 0)
def setScore(newScore: int) -> None    # Set score to specific value
```

###### Pause System

```python
def changePaused(player: Player) -> None
```

**Description:**
Toggles pause state and updates visual elements (colors, player sprite) to reflect the current state.

**Parameters:**
- `player` (Player): Player object to update sprite for

###### Seed Management

`getSeed()` / `setSeed(newSeed)` store the seed of the current episode. `setSeedVisible(bool)` / `isSeedVisible()` control whether the HUD shows it (`Simulation` enables it only for explicit seeds).

###### Pace Management

```python
def setPace(newPace: int) -> None
```

**Description:**
Manually set the game pace value.

**Parameters:**
- `newPace` (int): New pace value

---

### Clock (`clock.py`)

Game timing and frame rate management system.

#### Class: `Clock`

##### Constructor

```python
def __init__(self)
```

**Description:**
Initializes the clock with pygame's built-in clock and timing variables.

##### Methods

###### `update() -> None`

**Updates the clock and timing information.**

**Description:**
Called each frame to update current time, calculate deltas, and maintain timing state.

###### `reset() -> None`

**Resets all timing values to initial state.**

**Description:**
Used when starting a new game to ensure clean timing state.

###### Timing Accessors

```python
def getTicks() -> int              # Get total game ticks
def getMillis() -> int             # Get current milliseconds
def getPrevMillis() -> int         # Get previous frame milliseconds
def getSeconds() -> int            # Get current seconds
def getPrevSeconds() -> int        # Get previous frame seconds
def getTimeString() -> str         # Get formatted time string (MM:SS)
```

###### `rollbackMillis(amount: int) -> None`

**Rollback milliseconds for pause functionality.**

**Parameters:**
- `amount` (int): Milliseconds to subtract from current time

**Description:**
Used during pause to prevent time advancement while maintaining accurate timing.

---

## Entities Module (`infinite_maze.entities`)

### Player (`player.py`)

Player character representation and movement management.

#### Class: `Player`

##### Constructor

```python
def __init__(self, xPosition: int, yPosition: int, headless: bool = False)
```

**Parameters:**
- `xPosition` (int): Initial X coordinate
- `yPosition` (int): Initial Y coordinate  
- `headless` (bool): Whether to load graphics (default: False)

**Description:**
Creates a player object with specified starting position. Loads player sprite or creates a simple shape if in headless mode.

**Example:**
```python
# Create player at starting position
player = Player(80, 223)

# Create headless player for testing
test_player = Player(100, 200, headless=True)
```

##### Properties

```python
width = 10      # Player width in pixels
height = 10     # Player height in pixels
speed = 1       # Movement speed per frame
```

##### Methods

###### Position Management

```python
def setX(xPosition: int) -> None        # Set X coordinate
def setY(yPosition: int) -> None        # Set Y coordinate
def getX() -> int                       # Get current X coordinate
def getY() -> int                       # Get current Y coordinate
def getPosition() -> Tuple[int, int]    # Get (x, y) position tuple
```

###### Movement

```python
def moveX(units: int) -> None           # Move horizontally
def moveY(units: int) -> None           # Move vertically
```

**Parameters:**
- `units` (int): Distance to move (positive or negative)

**Description:**
Movement is multiplied by the player's speed value. Positive X moves right, positive Y moves down.

**Example:**
```python
player.moveX(1)   # Move right by speed pixels
player.moveX(-1)  # Move left by speed pixels
player.moveY(-1)  # Move up by speed pixels
```

###### Graphics Management

```python
def setCursor(image: str) -> None       # Set player sprite image
def getCursor() -> pygame.Surface       # Get current sprite surface
```

###### Utility Methods

```python
def getSpeed() -> int                   # Get movement speed
def getWidth() -> int                   # Get player width
def getHeight() -> int                  # Get player height
def reset(xPosition: int, yPosition: int) -> None  # Reset to new position
```

---

### Maze (`maze.py`)

Maze generation and wall management system.

#### Class: `Line`

Represents individual wall segments in the maze.

##### Constructor

```python
def __init__(self, xStart: int, yStart: int, xEnd: int, yEnd: int)
```

**Parameters:**
- `xStart, yStart` (int): Starting coordinates
- `xEnd, yEnd` (int): Ending coordinates

**Description:**
Creates a line segment representing a wall. Can be horizontal or vertical.

##### Methods

###### Position Accessors

```python
def getXStart() -> int                  # Get starting X coordinate
def getYStart() -> int                  # Get starting Y coordinate  
def getXEnd() -> int                    # Get ending X coordinate
def getYEnd() -> int                    # Get ending Y coordinate
def getStart() -> Tuple[int, int]       # Get starting point tuple
def getEnd() -> Tuple[int, int]         # Get ending point tuple
```

###### Position Modifiers

```python
def setXStart(x: int) -> None           # Set starting X coordinate
def setYStart(y: int) -> None           # Set starting Y coordinate
def setXEnd(x: int) -> None             # Set ending X coordinate
def setYEnd(y: int) -> None             # Set ending Y coordinate
```

###### Utility Methods

```python
def getIsHorizontal() -> bool           # Check if line is horizontal
```

**Returns:**
- `True` if the line is horizontal (yStart == yEnd)
- `False` if the line is vertical

##### Static Methods

###### `generateMaze(game: Game, rows: int, cols: int, rng: Optional[random.Random] = None) -> List[Line]`

**Generates a random maze layout.**

**Parameters:**
- `game` (Game): Game object for configuration
- `rows` (int): Number of maze rows
- `cols` (int): Number of maze columns
- `rng` (random.Random, optional): Source of randomness. The same seeded `rng` always yields the same maze; `None` uses a fresh unseeded generator (the global `random` module is never used).

**Returns:**
- `List[Line]`: List of line objects representing maze walls

**Description:**
Creates a procedurally generated maze using random wall placement. Ensures the maze is navigable while providing appropriate challenge.

**Example:**
```python
# Generate a 15x20 maze
lines = Line.generateMaze(game, 15, 20)
```

###### `getXMax(lines: List[Line]) -> int`

**Find the rightmost X coordinate among all lines.**

**Parameters:**
- `lines` (List[Line]): List of line objects

**Returns:**
- `int`: Maximum X coordinate found

**Description:**
Used for maze generation and line repositioning to maintain the infinite maze illusion.

---

## Utils Module (`infinite_maze.utils`)

### Config (`config.py`)

Centralized configuration management system.

#### Class: `GameConfig`

Central configuration class containing all game settings and constants.

##### Class Attributes

###### Display Settings
```python
SCREEN_WIDTH: int = 800             # Game window width
SCREEN_HEIGHT: int = 600            # Game window height  
FPS: int = 60                       # Target frames per second
```

###### Player Settings
```python
PLAYER_START_X: int = 80            # Player starting X position
PLAYER_START_Y: int = 223           # Player starting Y position
PLAYER_SPEED: int = 5               # Player movement speed
PLAYER_WIDTH: int = 20              # Player width in pixels
PLAYER_HEIGHT: int = 20             # Player height in pixels
```

###### Maze Settings
```python
MAZE_ROWS: int = 15                 # Default maze rows
MAZE_COLS: int = 20                 # Default maze columns
```

###### Color Definitions
```python
COLORS: Dict[str, Tuple[int, int, int]] = {
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255),
    "RED": (255, 0, 0),
    # ... more colors
}
```

###### Control Mappings
```python
CONTROLS: Dict[str, Any] = {
    "MOVE_RIGHT": ["RIGHT", "d"],
    "MOVE_LEFT": ["LEFT", "a"],
    "MOVE_UP": ["UP", "w"],
    "MOVE_DOWN": ["DOWN", "s"],
    "PAUSE": ["SPACE"],
    "QUIT": ["ESCAPE", "q"],
    "RESTART": ["r"],
}
```

###### Asset Paths
```python
IMAGES: Dict[str, str] = {
    "player": "assets/images/player.png",
    "player_paused": "assets/images/player_paused.png", 
    "icon": "assets/images/icon.png",
}
```

##### Class Methods

###### `get_color(color_name: str) -> Tuple[int, int, int]`

**Get RGB color tuple by name.**

**Parameters:**
- `color_name` (str): Name of the color

**Returns:**
- `Tuple[int, int, int]`: RGB color tuple

**Example:**
```python
red = GameConfig.get_color("red")      # Returns (255, 0, 0)
blue = GameConfig.get_color("blue")    # Returns (0, 0, 255)
```

###### `get_image_path(image_name: str) -> str`

**Get full path for an image asset.**

**Parameters:**
- `image_name` (str): Name of the image

**Returns:**
- `str`: Full path to the image file

**Example:**
```python
player_img = GameConfig.get_image_path("player")
# Returns "assets/images/player.png"
```

###### `get_asset_path(*path_parts: str) -> str`

**Build path relative to assets directory.**

**Parameters:**
- `*path_parts` (str): Path components to join

**Returns:**
- `str`: Full asset path

**Example:**
```python
sound_path = GameConfig.get_asset_path("sounds", "jump.wav")
# Returns "assets/sounds/jump.wav"
```

###### `get_movement_constant(action: str) -> int`

**Get movement constant by action name.**

**Parameters:**
- `action` (str): Action name

**Returns:**
- `int`: Movement constant value

**Example:**
```python
right_action = GameConfig.get_movement_constant("RIGHT")  # Returns 1
```

#### Default Config Instance

```python
config = GameConfig()
```

A default configuration instance available for import:

```python
from infinite_maze.utils.config import config
player_speed = config.PLAYER_SPEED
```

---

### Logger (`logger.py`)

Centralized logging system for debugging and monitoring.

#### Class: `GameLogger`

Provides structured logging capabilities for the game.

##### Constructor

```python
def __init__(self, name: str = "infinite_maze", level: str = "INFO")
```

**Parameters:**
- `name` (str): Logger name (default: "infinite_maze")
- `level` (str): Logging level (default: "INFO")

##### Methods

###### Basic Logging

```python
def debug(message: str) -> None         # Debug level messages
def info(message: str) -> None          # Info level messages  
def warning(message: str) -> None       # Warning level messages
def error(message: str) -> None         # Error level messages
def critical(message: str) -> None      # Critical level messages
```

###### Game-Specific Logging

```python
def log_game_start() -> None            # Log game initialization
def log_game_end(score: int) -> None    # Log game completion
def log_player_movement(direction: str) -> None  # Log player actions
def log_collision(details: str) -> None # Log collision events
def log_score_change(old: int, new: int) -> None  # Log score updates
```

#### Default Logger Instance

```python
logger = GameLogger()
```

Usage example:
```python
from infinite_maze.utils.logger import logger

logger.info("Game started")
logger.debug(f"Player position: {player.getPosition()}")
logger.warning("Performance slow")
```

---

## Constants & Enums

### Movement Constants

```python
DO_NOTHING = 0  # No movement action
RIGHT = 1       # Move right
LEFT = 2        # Move left  
UP = 3          # Move up
DOWN = 4        # Move down
```

### Game State Constants

```python
ACTIVE = "active"           # Game is running
PAUSED = "paused"          # Game is paused
GAME_OVER = "game_over"    # Game has ended
QUIT = "quit"              # Application should exit
```

---

## Usage Examples

### Basic Game Initialization

```python
from infinite_maze import Game, Player, Line

# Create game components
game = Game()
player = Player(80, 223)
maze_lines = Line.generateMaze(game, 15, 20)

# Game loop would go here
while game.isPlaying():
    # Handle input, update state, render
    game.updateScreen(player, maze_lines)
```

### Custom Configuration

```python
from infinite_maze.utils.config import GameConfig

# Create custom configuration
class CustomConfig(GameConfig):
    PLAYER_SPEED = 10      # Faster player
    SCREEN_WIDTH = 1024    # Larger screen
    SCREEN_HEIGHT = 768

# Use custom config
config = CustomConfig()
```

### Headless Mode for Testing

```python
from infinite_maze import Game, Player

# Create headless game for testing
game = Game(headless=True)
player = Player(80, 223, headless=True)

# Test game logic without graphics
assert player.getX() == 80
player.moveX(5)
assert player.getX() == 85
```

---

This API reference provides comprehensive documentation for extending and modifying the Infinite Maze game. For implementation examples and usage patterns, refer to the existing codebase and other documentation files.
