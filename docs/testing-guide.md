# Infinite Maze - Testing Guide

A comprehensive guide to the Infinite Maze test suite, covering test categories, running tests, writing new tests, and maintaining test quality.

## 📊 Test Suite Overview

The Infinite Maze project includes a professional-grade test suite with **293 test methods** across four categories, providing comprehensive coverage of all game components and functionality.

### Test Statistics
- **Total Test Methods**: 293
- **Unit Tests**: 184 test methods (63%)
- **Integration Tests**: 69 test methods (24%)
- **Functional Tests**: 21 test methods (7%)
- **Performance Tests**: 19 test methods (6%)

### Coverage Targets
- **Line Coverage**: >95% for new code
- **Branch Coverage**: >90% target
- **Function Coverage**: 100% target

---

## 🏗️ Test Architecture

### Test Structure
```
tests/
├── conftest.py                              # Pytest configuration and shared fixtures
├── pytest.ini                              # Test settings and markers
├── requirements-test.txt                    # Test dependencies
├── run_tests.py                            # Custom test runner
├── README.md                               # Test suite documentation
├── fixtures/                               # Test utilities and helpers
│   ├── __init__.py
│   ├── game_fixtures.py                   # Game-specific fixtures
│   ├── pygame_mocks.py                    # Pygame mocking framework
│   └── test_helpers.py                    # Common utilities
├── unit/                                   # Unit tests (184 tests)
│   ├── test_player.py                     # Player entity (34 tests)
│   ├── test_game.py                       # Game class (40 tests)
│   ├── test_clock.py                      # Clock/timing (28 tests)
│   ├── test_maze.py                       # Maze generation (40 tests)
│   └── test_config.py                     # Configuration (42 tests)
├── integration/                           # Integration tests (69 tests)
│   ├── test_player_maze_interaction.py    # Player-maze collision (20 tests)
│   ├── test_game_engine_integration.py    # Game engine integration (23 tests)
│   └── test_asset_loading_integration.py  # Asset loading (26 tests)
├── functional/                            # Functional tests (21 tests)
│   └── test_gameplay_scenarios.py         # End-to-end gameplay
└── performance/                           # Performance tests (19 tests)
    └── test_performance_benchmarks.py     # Performance monitoring
```

### Test Categories

#### 1. Unit Tests (`tests/unit/`)
Test individual components in isolation with comprehensive mocking:

**Player Tests** (34 tests):
- Movement mechanics and position tracking
- Collision bounds and boundary validation
- Sprite management and animation
- State persistence and restoration

**Game Tests** (40 tests):
- Game state management (active, paused, game over)
- Scoring system (increment, decrement, boundaries)
- Display coordination and rendering
- Game loop control and timing

**Clock Tests** (28 tests):
- Time tracking accuracy and FPS monitoring
- Time formatting and display utilities
- Rollback functionality for pause/resume
- Performance timing measurements

**Maze Tests** (40 tests):
- Line generation algorithms and boundary detection
- Collision detection accuracy and edge cases
- Maze structural integrity and solvability
- Performance optimization validation

**Config Tests** (42 tests):
- Configuration loading and validation
- Asset path resolution and verification
- Game settings management and defaults
- Environment-specific configuration handling

#### 2. Integration Tests (`tests/integration/`)
Test interactions between multiple components:

**Player-Maze Integration** (20 tests):
- Real-time collision detection between player and maze walls
- Movement constraints and boundary enforcement
- Spatial relationship validation and edge case handling

**Game Engine Integration** (23 tests):
- Coordination between game loop, player, clock, and maze
- State synchronization across components
- Resource management and cleanup

**Asset Loading Integration** (26 tests):
- Integration between configuration, asset loading, and game components
- Error handling for missing or corrupted assets
- Platform-specific asset path resolution

#### 3. Functional Tests (`tests/functional/`)
Test complete end-to-end workflows:

**Gameplay Scenarios** (21 tests):
- Complete game sessions from startup to game over
- User interaction simulation with realistic input patterns
- Pause/resume cycles and state persistence
- Error recovery and edge case scenarios

#### 4. Performance Tests (`tests/performance/`)
Test efficiency and establish performance benchmarks:

**Performance Benchmarks** (19 tests):
- Frame rate stability and consistency (60 FPS target)
- Memory usage monitoring and leak detection (<200MB target)
- Collision detection speed optimization (>1M checks/sec)
- Game loop efficiency measurement (<16.67ms/frame)

---

## 🚀 Running Tests

### Prerequisites
```bash
# Install test dependencies (one-time setup)
pip install -r requirements-test.txt
```

### Quick Testing (Development Workflow)

#### Using the Custom Test Runner
```bash
# Quick tests for daily development (unit + integration)
python run_tests.py

# Specific test categories
python run_tests.py unit                    # Unit tests only
python run_tests.py integration             # Integration tests only
python run_tests.py functional              # Functional tests only
python run_tests.py performance             # Performance tests only

# All tests including performance benchmarks
python run_tests.py all

# With coverage reporting
python run_tests.py unit --coverage
python run_tests.py all --coverage
```

#### Using pytest Directly
```bash
# Run all tests
pytest

# Run specific test categories using markers
pytest -m unit                              # Unit tests
pytest -m integration                       # Integration tests
pytest -m functional                        # Functional tests
pytest -m performance --run-performance     # Performance tests

# Run specific test files
pytest tests/unit/test_player.py
pytest tests/integration/test_player_maze_interaction.py

# Run tests matching a pattern
pytest -k "collision"                       # All collision-related tests
pytest -k "player_movement"                 # Player movement tests

# Skip slow tests during development
pytest --skip-slow
```

### Comprehensive Testing

#### Coverage Analysis
```bash
# Generate HTML coverage report
pytest --cov=infinite_maze --cov-report=html

# Terminal coverage report with missing lines
pytest --cov=infinite_maze --cov-report=term-missing

# XML coverage report (for CI/CD)
pytest --cov=infinite_maze --cov-report=xml
```

#### Performance Testing
```bash
# Run performance tests (disabled by default)
pytest -m performance --run-performance

# Performance tests with detailed output
pytest -m performance --run-performance -v

# Performance tests only
python run_tests.py performance
```

### Determinism, Golden Traces and Replays

The rules are tick based and the maze is seeded, so tests need no clock mocks:

- `tests/unit/test_simulation.py` drives `Simulation` directly (blocking, scoring, scrolling, pace at exactly `config.PACE_UPDATE_TICKS`, truncation, determinism over thousands of steps).
- `tests/unit/test_maze_analysis.py` verifies generated mazes are connected spanning trees and solvable.
- **Golden traces** (`tests/integration/test_golden_traces.py`) drive the real `maze()` / `controlled_run()` loops with scripted input (scenarios in `tests/fixtures/golden_scenarios.py`) and compare per-frame state with `tests/fixtures/golden/*.json`. A golden diff means gameplay changed. Regenerate **only** after an intentional change, and explain every diff in the PR:
  ```bash
  REGENERATE_GOLDENS=1 poetry run pytest tests/integration/test_golden_traces.py tests/unit/test_replay.py
  ```
- **Replay fixtures** in `tests/fixtures/replays/` (recipes in `tests/fixtures/replay_recipes.py`) are replayed headless and must match their `expected` state. The same environment variable regenerates them.
- Reproduce a bug from a recording: `poetry run python -m infinite_maze --replay bug.json --headless` (exit code 1 on mismatch).

### Optional ML dependencies

The `training/` package has tests (`tests/**/test_training_*.py`). Tests that need the optional Poetry groups call `pytest.importorskip`, so they are skipped when the libraries are missing and the core suite never needs them:

- `ml-env` (Gymnasium, numpy): `tests/unit/test_training_env.py`. CI runs it in its own job.
- `ml-train` (Stable-Baselines3, PyTorch): `tests/unit/test_training_train.py`. The end-to-end smoke run is marked `slow`: `poetry run pytest -m slow tests/ -k training`.
- Everything else under `training/` (observation, reward, evaluation, experiment configs, expert ladder, oracle) is stdlib-only and always runs.

### CI/CD Testing
```bash
# Fast feedback loop for pull requests
pytest -m "unit or integration" --skip-slow

# Full test suite for releases
pytest --run-performance --cov=infinite_maze --cov-report=xml
```

---

## ✍️ Writing Tests

### Test Structure Guidelines

#### Test Class Organization
```python
# tests/unit/test_new_component.py
class TestNewComponent:
    """Test class for NewComponent functionality."""
    
    def test_initialization(self):
        """Test component initialization with default values."""
        # Arrange
        component = NewComponent()
        
        # Act & Assert
        assert component.is_valid()
        assert component.get_default_value() == expected_default
        
    def test_specific_behavior(self):
        """Test specific behavior with descriptive name."""
        # Arrange
        component = NewComponent(param=value)
        
        # Act
        result = component.method_under_test()
        
        # Assert
        assert result == expected_result
```

#### Using Test Fixtures
```python
def test_with_game_fixture(headless_game, test_player):
    """Test using shared fixtures for consistent setup."""
    # Fixtures provide clean, isolated test environment
    game = headless_game  # Pre-configured game instance
    player = test_player  # Pre-positioned test player
    
    # Test implementation
    assert game.isActive()
    assert player.getPosition() == (80, 223)
```

### Test Categories by Change Type

#### New Component Tests
When adding a new component, create unit tests:
```python
# tests/unit/test_new_component.py
class TestNewComponent:
    def test_initialization(self):
        """Test component initializes correctly."""
        pass
        
    def test_core_functionality(self):
        """Test primary component behavior."""
        pass
        
    def test_edge_cases(self):
        """Test boundary conditions and error cases."""
        pass
        
    def test_integration_points(self):
        """Test interfaces with other components."""
        pass
```

#### Feature Integration Tests
When adding features involving multiple components:
```python
# tests/integration/test_new_feature_integration.py
class TestNewFeatureIntegration:
    def test_component_coordination(self):
        """Test components work together correctly."""
        pass
        
    def test_state_synchronization(self):
        """Test state consistency across components."""
        pass
```

#### End-to-End Feature Tests
When adding user-facing features:
```python
# tests/functional/test_new_feature_scenarios.py
class TestNewFeatureScenarios:
    def test_complete_workflow(self):
        """Test complete user workflow with new feature."""
        pass
        
    def test_user_interaction_patterns(self):
        """Test realistic user interaction scenarios."""
        pass
```

#### Performance Impact Tests
When making performance-related changes:
```python
# tests/performance/test_new_performance_aspects.py
@pytest.mark.performance
class TestNewPerformanceAspects:
    def test_performance_baseline(self):
        """Establish performance baseline for new functionality."""
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Performance test implementation
        
        monitor.stop()
        assert monitor.get_duration() < acceptable_threshold
        assert monitor.get_memory_usage() < memory_limit
```

### Test Quality Standards

#### Test Naming Conventions
- **Test files**: `test_<component_name>.py`
- **Test classes**: `Test<ComponentName>`
- **Test methods**: `test_<specific_behavior>`
- **Descriptive names**: Clearly describe what is being tested

#### Test Documentation
```python
def test_player_collision_boundary_detection(self):
    """Test player collision detection at maze boundaries.
    
    Verifies that collision detection correctly identifies when the player
    is at the exact boundary of a maze wall, including edge cases where
    the player position aligns precisely with wall coordinates.
    """
    # Test implementation
```

#### Assertion Guidelines
- **Single concept**: Each test should verify one specific behavior
- **Clear assertions**: Use descriptive assertion messages
- **Edge cases**: Test boundary conditions and error scenarios
- **Deterministic**: Tests should produce consistent results

#### Example Test Implementation
```python
class TestPlayerMovement:
    def test_move_right_updates_position(self):
        """Test that moving right increases X coordinate by movement speed."""
        # Arrange
        player = Player(100, 200, headless=True)
        initial_x = player.getX()
        
        # Act
        player.moveX(5)  # Move right by 5 pixels
        
        # Assert
        assert player.getX() == initial_x + 5
        assert player.getY() == 200  # Y should remain unchanged
        
    def test_collision_detection_prevents_wall_penetration(self):
        """Test that collision detection prevents player from moving through walls."""
        # Arrange
        player = Player(95, 200, headless=True)
        wall_line = Line(100, 200, 100, 250)  # Vertical wall at x=100
        
        # Act
        player.moveX(10)  # Attempt to move through wall
        collision_detected = wall_line.collision(player)
        
        # Assert
        assert collision_detected == True
        # Player position should be constrained by collision detection
```

---

## 🔧 Test Infrastructure

### Pygame Mocking Framework

The test suite uses comprehensive pygame mocking to enable headless testing:

#### Key Features
- **No Display Required**: Tests run without actual pygame windows
- **Deterministic Input**: Controlled event simulation
- **Performance Isolation**: Clean performance measurement
- **CI/CD Compatible**: Runs in automated environments

#### Using Mocked Components
```python
def test_with_pygame_mocking(mock_pygame_display):
    """Test using pygame mocking for headless execution."""
    # Pygame display operations are automatically mocked
    game = Game(headless=True)
    
    # All pygame calls are intercepted and handled by mocks
    assert game.initialize() == True
```

### Test Fixtures

#### Core Fixtures (`conftest.py`)
```python
@pytest.fixture
def headless_game():
    """Provide a clean game instance for testing."""
    return Game(headless=True)

@pytest.fixture  
def test_player():
    """Provide a test player at standard position."""
    return Player(80, 223, headless=True)

@pytest.fixture
def test_maze():
    """Provide a test maze with known layout."""
    return generate_test_maze()
```

#### Specialized Fixtures (`fixtures/game_fixtures.py`)
```python
@pytest.fixture
def collision_test_scenarios():
    """Provide predefined collision test scenarios."""
    return [
        {"player_pos": (100, 200), "wall": Line(105, 200, 105, 250), "should_collide": True},
        {"player_pos": (90, 200), "wall": Line(105, 200, 105, 250), "should_collide": False},
        # Additional test scenarios
    ]
```

### Performance Monitoring

#### Performance Test Utilities
```python
class PerformanceMonitor:
    """Monitor performance metrics during testing."""
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.get_memory_usage()
        
    def stop(self):
        """Stop monitoring and calculate metrics."""
        self.duration = time.time() - self.start_time
        self.peak_memory = self.get_memory_usage()
        
    def get_duration(self) -> float:
        """Get execution duration in seconds."""
        return self.duration
        
    def get_memory_usage(self) -> int:
        """Get memory usage in MB."""
        # Implementation for memory monitoring
```

#### Using Performance Monitoring
```python
@pytest.mark.performance
def test_collision_detection_performance(self):
    """Test collision detection meets performance targets."""
    monitor = PerformanceMonitor()
    player = Player(100, 200, headless=True)
    walls = generate_performance_test_maze(1000)  # 1000 walls
    
    monitor.start()
    
    # Perform 1 million collision checks
    for _ in range(1000000):
        for wall in walls:
            wall.collision(player)
    
    monitor.stop()
    
    # Verify performance targets
    checks_per_second = 1000000 / monitor.get_duration()
    assert checks_per_second > 1000000  # >1M checks/second target
    assert monitor.get_memory_usage() < 200  # <200MB memory target
```

---

## 📈 Test Maintenance

### Keeping Tests Updated

#### When Code Changes
1. **Run affected tests**: Identify and run tests related to changed code
2. **Update test data**: Modify test fixtures if interfaces change
3. **Add new tests**: Create tests for new functionality
4. **Update documentation**: Keep test documentation current

#### Regular Maintenance Tasks
```bash
# Weekly: Run full test suite
python run_tests.py all --coverage

# Monthly: Review and update performance baselines
python run_tests.py performance

# Before releases: Comprehensive testing
pytest --run-performance --cov=infinite_maze --cov-report=html
```

### Test Quality Metrics

#### Coverage Monitoring
```bash
# Check current coverage
pytest --cov=infinite_maze --cov-report=term-missing

# Identify untested code
pytest --cov=infinite_maze --cov-report=html
# Open htmlcov/index.html to see detailed coverage report
```

#### Performance Baseline Updates
Performance baselines should be updated when:
- **Optimization improvements** are made
- **System requirements** change  
- **New features** affect performance characteristics
- **Hardware upgrades** improve baseline capabilities

### Adding Tests for New Features

#### Step-by-Step Process
1. **Identify test category**: Unit, integration, functional, or performance
2. **Choose appropriate location**: Place in correct `tests/` subdirectory
3. **Use existing patterns**: Follow established test structure
4. **Add appropriate markers**: Use `@pytest.mark.unit`, etc.
5. **Update documentation**: Add test descriptions to this guide

#### Example: Adding Tests for New Game Mode
```python
# tests/unit/test_game_modes.py
class TestGameModes:
    def test_classic_mode_initialization(self):
        """Test classic game mode initializes correctly."""
        game = Game(mode="classic", headless=True)
        assert game.get_mode() == "classic"
        assert game.get_pace_interval() == 30
        
    def test_speed_mode_initialization(self):
        """Test speed game mode has faster pace."""
        game = Game(mode="speed", headless=True)
        assert game.get_mode() == "speed"
        assert game.get_pace_interval() == 15  # Faster pace

# tests/functional/test_game_mode_scenarios.py  
class TestGameModeScenarios:
    def test_classic_mode_complete_gameplay(self):
        """Test complete gameplay session in classic mode."""
        # End-to-end test implementation
        
    def test_mode_switching_preserves_score(self):
        """Test that switching modes preserves player progress."""
        # Mode switching test implementation
```

---

## 🚨 Troubleshooting Tests

### Common Test Issues

#### Import Errors
```bash
# Problem: ModuleNotFoundError when running tests
# Solution: Install project in development mode
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Pygame Display Errors
```bash
# Problem: "pygame.error: video system not initialized"
# Solution: Ensure headless mode is enabled
# Tests should use headless=True by default
```

#### Performance Test Failures
```bash
# Problem: Performance tests fail on slower systems
# Solution: Skip performance tests during development
pytest --skip-slow

# Or run performance tests in isolation
pytest -m performance --run-performance
```

#### Coverage Calculation Issues
```bash
# Problem: Coverage reports show incorrect data
# Solution: Install coverage tools explicitly
pip install pytest-cov

# Run with explicit coverage configuration
pytest --cov=infinite_maze --cov-report=html
```

### Debugging Test Failures

#### Verbose Test Output
```bash
# Run tests with detailed output
pytest -v

# Show print statements during tests
pytest -s

# Debug specific test failure
pytest tests/unit/test_player.py::TestPlayerMovement::test_move_right -v -s
```

#### Test Isolation
```bash
# Run single test file
pytest tests/unit/test_player.py

# Run single test method
pytest tests/unit/test_player.py::TestPlayerMovement::test_move_right

# Run tests matching pattern
pytest -k "collision and not performance"
```

---

## 🎯 Best Practices

### Test Development Guidelines

#### 1. Test-Driven Development (TDD)
- Write tests before implementing features
- Use tests to define expected behavior
- Refactor with confidence knowing tests catch regressions

#### 2. Maintainable Test Code
- Keep tests simple and focused
- Use descriptive names and documentation
- Avoid duplicating production code logic in tests
- Use fixtures to reduce test setup complexity

#### 3. Comprehensive Coverage
- Test happy paths and edge cases
- Include error conditions and boundary values
- Test component interactions and integration points
- Verify performance characteristics don't degrade

#### 4. Fast Feedback Loop
- Run relevant tests frequently during development
- Use test markers to run focused test subsets
- Keep unit tests fast (<100ms each)
- Use mocking to isolate component dependencies

### Performance Testing Best Practices

#### 1. Baseline Establishment
- Establish performance baselines for critical operations
- Document acceptable performance ranges
- Update baselines when optimizations are made
- Monitor performance trends over time

#### 2. Realistic Test Scenarios
- Use realistic data sizes and complexity
- Test under various load conditions
- Include worst-case scenarios in testing
- Validate performance on target hardware configurations

#### 3. Measurement Accuracy
- Run performance tests multiple times for statistical significance
- Account for system variability and background processes
- Use dedicated performance test environments when possible
- Isolate performance tests from functional tests

---

## 📚 Additional Resources

### Documentation Links
- **[Main README](../README.md)**: Project overview and quick start
- **[Contributing Guide](contributing.md)**: Development workflow and contribution guidelines
- **[Architecture Guide](architecture.md)**: Technical architecture and testing infrastructure
- **[API Reference](api-reference.md)**: Component interfaces and testing points

### External Resources
- **[Pytest Documentation](https://docs.pytest.org/)**: Comprehensive pytest guide
- **[Coverage.py Documentation](https://coverage.readthedocs.io/)**: Code coverage measurement
- **[Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)**: General Python testing guidelines

### Test Suite Statistics Summary
- **Total Tests**: 293 methods across 10 test files
- **Unit Tests**: 184 tests (63% of total)
- **Integration Tests**: 69 tests (24% of total)  
- **Functional Tests**: 21 tests (7% of total)
- **Performance Tests**: 19 tests (6% of total)
- **Target Coverage**: >95% line coverage
- **Performance Targets**: 60 FPS, <200MB memory, >1M collision checks/sec

---

*This testing guide is maintained alongside the test suite to ensure accuracy and completeness. For questions about testing or to report issues with the test suite, please refer to the main project documentation or open an issue on GitHub.*
