---
description: "Use when writing, editing, or reviewing test files for the Infinite Maze repo. Covers fixture selection, pygame mock usage, headless test patterns, and test layer conventions."
applyTo: "tests/**/*.py"
---
# Infinite Maze — Test Guidelines

## Headless-Safe Rule

Every test must run without a display. Never call `pygame.display.set_mode()`, `pygame.display.flip()`, or load real image files directly in test code — always use the fixtures and mocks below.

## CI Alignment

- Run tests through Poetry so imports and dependencies match CI.
- If pytest plugins are missing, install test deps into Poetry's env with:
    - `poetry run pip install -r requirements-test.txt`
- Default CI selection is:
    - `poetry run pytest -m "not slow and not performance"`

## Fixture Reference

### `conftest.py` fixtures (auto-available to all tests)

| Fixture | Scope | Use for |
|---|---|---|
| `pygame_init` | session | Always-on; initialises and tears down pygame for the whole session — do not call `pygame.init()` manually |
| `mock_pygame_display` | function | Tests that run through rendering code; patches `set_mode`, `flip`, `set_caption`, `set_icon`; yields a dict with `surface`, `flip`, etc. |
| `mock_pygame_image` | function | Tests that construct `Player` or any code that calls `pygame.image.load()` |
| `mock_pygame_font` | function | Tests that exercise HUD/overlay text rendering |
| `mock_pygame_time` | function | Tests that need controlled clock timing |

### `tests/fixtures/pygame_mocks.py` — low-level mock classes

| Class | Use for |
|---|---|
| `MockPygameSurface(size)` | Inspecting `fill()` and `blit()` calls; asserts on `surface.fills` / `surface.blits` lists |
| `MockPygameClock` | Controlled per-frame timing; `tick()` advances `time_value` by `frame_time` ms; check `ticks` count |
| `MockPygameFont(name, size)` | Asserting on rendered text; rendered calls stored in `font.rendered_texts` list |
| `MockPygameEvent(type, **kwargs)` | Injecting input events; pass instances into mocked `pygame.event.get()` returns |
| `PygameMockContext(...)` | Comprehensive context manager that patches display, time, font, image, event, and draw all at once; prefer this for integration-layer tests |

### `tests/fixtures/game_fixtures.py` — game-level helpers

| Class / method | Use for |
|---|---|
| `MockGameEngine` | Integration tests; call `setup_game(headless=True)` to get a real `Game` + `Player` + generated `Line` list without a window |
| `.simulate_movement(direction, frames)` | Moving the player N frames in `'right'`/`'left'`/`'up'`/`'down'` |
| `.get_collision_state()` | Getting a `dict[str, bool]` of which directions are currently blocked |

### `tests/fixtures/test_helpers.py` — assertion utilities

| Function | Use for |
|---|---|
| `assert_within_tolerance(actual, expected, tolerance)` | Float comparisons |
| `assert_position_equal(actual, expected, tolerance=1)` | Position tuple assertions with pixel tolerance |
| `assert_position_valid(position, bounds)` | Bounds-checking positions against `{'x_min', 'x_max', 'y_min', 'y_max'}` |
| `is_collision_detected(player, line)` | Manual AABB check between a `Player` and a `Line`; mirrors engine logic |
| `get_valid_moves(player, lines)` | Returns list of unblocked directions given current maze state |
| `calculate_distance(pos1, pos2)` | Euclidean distance between two `(x, y)` tuples |

## Test Layer Conventions

| Layer | Folder | Scope |
|---|---|---|
| Unit | `tests/unit/` | Single class or function; use conftest fixtures + mock classes; no `MockGameEngine` |
| Integration | `tests/integration/` | Cross-module behaviour; use `MockGameEngine` + `PygameMockContext` |
| Functional | `tests/functional/` | End-to-end gameplay scenarios; always fully headless via `MockGameEngine` |
| Performance | `tests/performance/` | Timing benchmarks; use `MockPygameClock` for controlled frames; compare against config-based thresholds |

## Timing in Tests

- Use `MockPygameClock` to control elapsed time — never use `time.sleep()` or `datetime.now()`.
- `MockPygameClock.tick()` advances `time_value` by `frame_time` ms (default 16 ms / 60 fps). Override `frame_time` on the instance to simulate slower frames.
- For pace-system tests, calculate the number of `Simulation.step()` calls needed from `config.PACE_UPDATE_TICKS` rather than hardcoding frame counts.

## Common Patterns

**Constructing a headless Player:**
```python
player = Player(80, 223, headless=True)
```

**Constructing a headless Game + maze:**
```python
engine = MockGameEngine()
engine.setup_game(headless=True)
# engine.game, engine.player, engine.lines now available
```

**Asserting no real display was opened:**
```python
def test_something(mock_pygame_display):
    # ... exercise code ...
    mock_pygame_display['set_mode'].assert_called_once()
```

**Injecting a key-press event:**
```python
event = MockPygameEvent(pygame.KEYDOWN, key=pygame.K_RIGHT)
with patch('pygame.event.get', return_value=[event]):
    # ... run one engine frame ...
```
