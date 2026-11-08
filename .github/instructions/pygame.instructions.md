---
description: "Use when writing, editing, or reviewing any Python source in the Infinite Maze repo. Covers pygame patterns, code conventions, config usage, entity structure, testing rules, and collision/rendering guidelines."
applyTo: "infinite_maze/**/*.py"
---
# Infinite Maze — Python/Pygame Coding Guidelines

## Code Conventions

- Use **getter/setter methods** for all entity properties — `getX()`, `setX()`, `getIsHorizontal()`, etc. Do not use bare attribute access or Python `@property` decorators on existing entity classes.
- All numeric constants (screen size, speeds, intervals, tile sizes) must come from `utils/config.py`. Never hardcode magic numbers in source files.
- Keep entity classes (`Player`, `Line`) as focused data holders. Game-logic coordination belongs in `core/engine.py`, not in the entity itself.
- New config values go in `utils/config.py` first, then reference them everywhere else.

## Pygame Patterns

- **Game loop**: Drain the event queue each frame with `for event in pygame.event.get()`, then poll held keys with `pygame.key.get_pressed()`. Call `clock.tick()` once per frame at the end.
- **Presenting frames**: Always use `pygame.display.flip()` — do not use `pygame.display.update()`.
- **Timing**: Use the custom `Clock` wrapper in `core/clock.py`. Do not instantiate `pygame.time.Clock` directly in new code.
- **Drawing maze walls**: Use `pygame.draw.line(surface, color, start_pos, end_pos, width)` — match the call signature in the existing render loop.
- **Sprites**: Load via `pygame.image.load()` and convert with `.convert_alpha()` where transparency is needed. Always provide a solid-colour fallback surface (red square) if the asset path is missing.
- **Fonts**: Use `pygame.font.SysFont()` for HUD/overlay text. Keep font creation outside the game loop to avoid per-frame allocation.
- **Surfaces**: Create off-screen surfaces with `pygame.Surface((w, h))`, not with `pygame.display.set_mode()`.

## Collision

- Use **AABB (axis-aligned bounding box)** intersection for player-vs-wall checks. Represent the player as a `pygame.Rect`; `Line` walls expose start/end tuples and a horizontal flag — derive a thin rect from them for the check.
- Do not use `pygame.sprite.Group` or `pygame.sprite.collide_*` — the codebase uses manual rect-based collision; keep it consistent.

## Clock & Pause Logic

- The `Clock` wrapper tracks paused time separately. Call `clock.rollbackMillis()` when exiting pause to avoid jump-cuts in movement or pace calculations.
- Check `game.isPaused()` before advancing pace or updating positions — never tick game logic while paused.

## Pace System

- The global speed multiplier lives in `Game` and increments every `config.PACE_UPDATE_TICKS` simulation ticks (a frame count, not wall-clock time); the logic lives in `Simulation.step()`.
- Maze lines reposition off-screen to the right as pace increases — do not reset their absolute coordinates to zero.
- If adding a new speed-dependent value, multiply by the current pace factor and source the base value from `config`.

## Testing Rules

- All tests must be **headless-safe**: mock `pygame.display`, `pygame.event`, and surface creation using the fixtures in `tests/fixtures/pygame_mocks.py`. Never call `pygame.display.set_mode()` in test code.
- Use `pytest`; place unit tests in `tests/unit/`, integration tests in `tests/integration/`.
- Do not add `time.sleep()` or real timers in tests — use the `Clock` mock to control elapsed time.
- Keep local checks aligned with CI before finishing source changes:
	- `poetry run black --check infinite_maze/`
	- `poetry run flake8 infinite_maze/`
	- `poetry run mypy infinite_maze/`
	- `poetry run pytest -m "not slow and not performance"`
- If pytest tools are missing in Poetry's env, install them with:
	- `poetry run pip install -r requirements-test.txt`

## Dependencies

- Do not add packages outside `pygame` and the Python standard library without explicit user approval.
- If a stdlib module would suffice, prefer it over a third-party package.
