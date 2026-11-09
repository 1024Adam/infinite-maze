# Plan 1 — Foundations and Debug Tooling (merge to `main`)

Status: proposed
Scope: seedable maze, characterization tests, simulated time, a single `step()` implementation, episode caps, record/replay, maze solvability checks.
Audience: implemented by the repo owner (or the `pygame-dev` agent) and merged to `main` ahead of any ML work.
Follow-up: [Plan 2 — ML debug aids](./02-ml-debug-aids.md) (dev branch only).

## 1. Goals and non-goals

**Goals**

- Make any game reproducible from `(seed, action sequence)`.
- Have exactly one implementation of the game rules (movement, collision, scroll, pace, death), shared by human play, `controlled_run`, tests and (later) an RL environment.
- Remove wall-clock dependence from the rules, so tests are fast and deterministic.
- Add debugging tools useful to any developer: seed flag, seed in HUD, record/replay, maze verification.

**Non-goals**

- No ML dependencies, no Gymnasium, no observation/reward code (that is Plan 2 and later).
- No change to human gameplay feel, other than the small deltas listed in section 4 (all explicitly called out).
- No performance rewrite of collision or maze generation. Profile first; it is out of scope here.
- Core dependencies stay `pygame` + stdlib. Code must remain Python 3.8 compatible (`Optional[...]`, not `X | None`).

## 2. Current state (verified by reading the code)

| Area | File | Finding |
|---|---|---|
| Maze RNG | [entities/maze.py](../infinite_maze/entities/maze.py) | `generateMaze` uses the global `randint`; no seed or RNG injection. Called with positional `(game, width, height)` in ~40 existing tests, so any new parameter must be optional and last. |
| Cell IDs | same | Cell IDs use a hard-coded stride of `19` (`sideA = 19*x + 1`, `sideB = sideA + 19`). That equals `height - 1` for the default config (`height = 20`), but would collide for `height > 20`. Unverified for larger sizes; PR 1 tests it. |
| Duplicated rules | [core/engine.py](../infinite_maze/core/engine.py) | `maze()` and `controlled_run()` each contain ~150 lines of inline AABB checks, scroll, bounds and recycle logic. |
| Pace / time | [core/game.py](../infinite_maze/core/game.py) | Pace increments in `updateScreen()` based on wall-clock seconds (`PACE_UPDATE_INTERVAL = 30`). The logic is duplicated in the headless and rendered branches. |
| Scroll | engine | Scroll occurs when `clock.getTicks() % 10 == 0`. Ticks are incremented in `Clock.update()` at the start of each frame (before movement), including paused frames. |
| FPS limiting | engine | `time.delay(FPS_DELAY_MS - processTime)` is inside the game loop. |
| Human input | engine | Horizontal (`RIGHT`/`LEFT`, exclusive) and vertical (`DOWN`/`UP`, exclusive) are evaluated independently, so diagonal movement is possible. |
| `controlled_run` | engine | One action per frame, from the wrapper. Keyboard input overrides the action if any key is pressed. Returns state to `wrapper.control({"action","score_increased","closest_line"})`. |
| Restart | engine | On restart, `player.reset(PLAYER_START_X, game.HEIGHT / 2)` (y = 240), whereas initial spawn uses `PLAYER_START_Y` (223). |
| Entry point | [pyproject.toml](../pyproject.toml) | `infinite-maze = "infinite_maze.core.engine:maze"` (no argument parsing). [\_\_main\_\_.py](../infinite_maze/__main__.py) just calls `maze()`. |
| Logging | [utils/logger.py](../infinite_maze/utils/logger.py) | Module-level `info()/debug()` helpers exist; use them for the seed log line. |

Observations to verify (not assumed true):

1. **Possible wall tunnelling at pace >= 2.** Several collision checks use `==` against a wall coordinate (for example right-moving vs horizontal lines). When lines shift `pace` pixels at a time, the equality may be stepped over. Characterization tests (PR 2) should include pace 2 and 3 scenarios and record the actual behaviour, then we decide whether it is a bug.
2. **The world looks periodic.** Recycled lines move to `xMax`, so the world appears to repeat every `2 * MAZE_ROWS` columns. This matters for the solvability check and later for ML, so confirm it in PR 1.

## 3. Target design

```text
                     +-------------------------------+
 human keys  ------> |  engine.maze()                |   rendering, input,
 wrapper.control --> |  engine.controlled_run()      |   FPS delay, pause   (adapters)
 replay file  -----> |  replay.run()                 |
                     +---------------+---------------+
                                     | Intent (dx, dy)
                                     v
                     +-------------------------------+
                     |  core/simulation.py           |   rules only: no sleep,
                     |  Simulation.step(intent)      |   no wall clock, no display
                     +---------------+---------------+
                                     |
              Line.generateMaze(..., rng=random.Random(seed))
```

### 3.1 `core/simulation.py` (new)

```python
Intent = Tuple[int, int]            # (dx, dy), each in {-1, 0, 1}

@dataclass
class StepResult:
    score_delta: int                # actual change to game score (floor at 0 preserved)
    moved: Tuple[int, int]          # actual player displacement from input, before scrolling
    terminated: bool                # pushed past X_MIN
    truncated: bool                 # hit max_steps

class Simulation:
    def __init__(self, seed: Optional[int] = None,
                 start_pace: int = config.START_PACE,
                 max_steps: Optional[int] = config.MAX_EPISODE_STEPS,
                 game: Optional[Game] = None,
                 player: Optional[Player] = None) -> None: ...
    def reset(self, seed: Optional[int] = None) -> None: ...
    def step(self, intent: Intent) -> StepResult: ...
    def getSeed(self) -> int: ...
    def getTicks(self) -> int: ...
    def getLines(self) -> List[Line]: ...
    def getClosestLineX(self) -> int: ...
```

- Entities keep their getter/setter style. Game-logic coordination stays in `core/` (per `.github/instructions/pygame.instructions.md`).
- `Simulation` uses an injected or headless `Game` for score, pace and bounds, so existing HUD and getters continue to work. `Game` still imports pygame and calls `pygame.init()` even when headless; removing that is explicitly *not* part of this plan.
- `step()` order must match today's frame order exactly:
  1. `ticks += 1` and pace update (formerly in `updateScreen`)
  2. horizontal move, then vertical move (each checks collision)
  3. scroll by `pace` if `ticks % SCROLL_INTERVAL_TICKS == 0`
  4. `player.x < X_MIN` means terminated
  5. `player.x > X_MAX` means clamp and scroll by `speed`
  6. clamp Y to `[Y_MIN, Y_MAX]`
  7. recycle lines whose `xStart < PLAYER_START_X`
  8. if `max_steps` is set and reached, `truncated = True`
- Collision is extracted into per-direction helpers (`_isBlocked(direction)`), replacing the four inline blocks per loop. **Keep the exact comparison semantics** (including the `==` checks) in PR 3; any fix is a separate, explicit PR.
- Discrete actions map to intents through a table in config, so there is one definition:

```python
ACTION_TO_INTENT: Dict[int, Tuple[int, int]] = {
    0: (0, 0),   # DO_NOTHING
    1: (1, 0),   # RIGHT
    2: (-1, 0),  # LEFT
    3: (0, -1),  # UP
    4: (0, 1),   # DOWN
}
```

- Human input builds an `Intent` directly, which preserves diagonals.

### 3.2 Seed handling

- `Line.generateMaze(game, width, height, rng: Optional[random.Random] = None)`. When `rng is None`, create a fresh `random.Random()`; the global `random` module is no longer consulted.
- The seed is always known. If none is supplied: `seed = random.SystemRandom().randrange(2**32)`. It is logged (`logger.info`) and shown in the HUD.
- Restart semantics (decision D1): an **explicit** seed regenerates the same maze on restart (practice mode); an **unspecified** seed gets a new random seed each episode.
- Public signatures stay backward compatible: `maze(seed: Optional[int] = None)`, `controlled_run(wrapper, counter, seed: Optional[int] = None)`.

### 3.3 Simulated time

New config values (`utils/config.py`):

```python
START_PACE: int = 0
SCROLL_INTERVAL_TICKS: int = 10
PACE_UPDATE_TICKS: int = PACE_UPDATE_INTERVAL * FPS   # 30 * 60 = 1800
MAX_EPISODE_STEPS: Optional[int] = None               # None = unlimited (human play)
ACTION_TO_INTENT: Dict[int, Tuple[int, int]] = {...}
```

- Pace increments when `ticks % PACE_UPDATE_TICKS == 0`.
- Remove pace logic from both branches of `Game.updateScreen()`; `updateScreen()` becomes render + clock only.
- `Clock` is kept for the HUD timer and FPS limiting only. Its public API does not change.
- `time.delay` and FPS limiting live only in the human/replay loops, never in `Simulation.step()`.

### 3.4 Record / replay

Format (JSON, versioned):

```json
{
  "version": 1,
  "seed": 1234,
  "start_pace": 0,
  "maze": {"rows": 15, "cols": 20, "cell": 22},
  "intents": [[1, 0, 120], [0, 1, 7], [1, 0, 30]],
  "expected": {"score": 85, "ticks": 157, "pace": 0, "player": [101, 223], "terminated": false}
}
```

- `intents` is run-length encoded as `[dx, dy, count]`. Only steps that actually reach `Simulation.step()` are recorded, so pauses are naturally excluded.
- `expected` is optional but is what makes a replay a regression test.
- New module `core/replay.py`: `ReplayRecorder`, `loadReplay(path)`, `runReplay(replay, render: bool) -> Dict[str, Any]`.
- `infinite_maze/__main__.py` gets argparse:

```text
infinite-maze [--seed N] [--start-pace N] [--record PATH] [--replay PATH] [--headless]
```

  - `--headless` is valid only with `--replay`. It prints the final state and exits non-zero if `expected` does not match.
  - Multiple episodes in one session write `PATH`, then `PATH` with `-2`, `-3` ... inserted before the extension.
- `pyproject.toml` script changes to `infinite_maze.__main__:main`. `maze()` remains importable and callable with no args.
- Local recordings go to `replays/` (add to `.gitignore`). Committed regression replays go to `tests/fixtures/replays/`.

### 3.5 Maze verification (`entities/maze_analysis.py`, new)

Pure functions on a freshly generated, unscrolled line list:

- `buildCellGraph(lines, width, height) -> Dict[Cell, Set[Cell]]` derived from geometry (a wall exists between two cells iff a line occupies their shared edge). Do not reuse `sideA`/`sideB`, so the check is independent of the generator's bookkeeping.
- `isConnected(graph) -> bool`
- `shortestPathLength(graph, startCell, goalColumn) -> Optional[int]`
- Expected property of the current algorithm (it only removes walls between different sets): the cell graph is a spanning tree, so edges = cells − 1. Assert this per seed.

## 4. Behaviour deltas to accept (or reject) explicitly

| # | Delta | Source | Recommendation |
|---|---|---|---|
| D1 | Explicit seed repeats the same maze on restart | new feature | Accept |
| D2 | Pace advances every 1800 ticks instead of every 30 real seconds. At ~62.5 FPS (16 ms delay), pace arrives ~4% earlier (~28.8 s) | simulated time | Accept; document. Alternative: derive the constant from `1000 / FPS_DELAY_MS` instead of `FPS = 60`. Pick one in PR 4. |
| D3 | Paused frames no longer advance `ticks` | `step()` is not called while paused | Accept (matches the "never tick logic while paused" guideline) |
| D4 | Restart spawn Y changes from 240 to `PLAYER_START_Y` (223) | single reset path | Accept; the inconsistency looks accidental |
| D5 | Possible tunnelling at pace >= 2 (see section 2) | characterization | Decide after PR 2 results; fix in its own PR, not inside the refactor |

## 5. Incremental delivery (each PR keeps CI green)

CI parity for every PR:

```bash
poetry run black --check infinite_maze/
poetry run flake8 infinite_maze/
poetry run mypy infinite_maze/
poetry run pytest -m "not slow and not performance"
```

### PR 1 — Seedable maze, CLI `--seed`, solvability tests (items 1, 9, part of 2)

Changes:

- `Line.generateMaze(..., rng=None)`; thread `seed` through `maze()` and `controlled_run()` (all three generation call sites: initial, restart, controlled run).
- `Game.getSeed()/setSeed()`; HUD line `Seed: N` placed where it does not overlap the `Y_MIN` border; `logger.info` at episode start.
- `__main__.py` argparse with `--seed`; update the `pyproject.toml` script entry.
- New `entities/maze_analysis.py`.

Tests:

- `tests/unit/test_maze.py`: same seed gives identical line lists; different seeds differ; `rng=None` still works; call without the new parameter is unchanged.
- `tests/unit/test_maze_analysis.py`: for seeds `0..N` and a few sizes, graph is connected and a spanning tree; shortest path exists from start to the right edge; periodicity check (confirm or refute the observation in section 2).
- Check `height > 20` explicitly. If it fails because of the hard-coded stride `19`, replace it with a stride derived from `height` in this PR (it is a bug tightly coupled to the verification work).
- `tests/unit/test_main_cli.py`: argument parsing only (no pygame window).

Acceptance: `infinite-maze --seed 1234` produces the same maze on every run; unseeded play unchanged.

### PR 2 — Characterization tests (item 2)

Goal: lock current behaviour *before* touching the loops.

- Harness in `tests/fixtures/` that drives the legacy `maze()` and `controlled_run()` under mocks (use `PygameMockContext`; patch `pygame.key.get_pressed`, `time.delay`, and `Game.updateScreen` with a stub that calls `clock.tick()` and sets the pace from a script, so runs are independent of wall time).
- Record per-frame `(x, y, score, pace, short hash of sorted line coordinates)`.
- Scenarios, using fixed seeds: walk right until blocked; slide along a wall; left movement at score 0 (floor); scroll at `ticks % 10`; recycling at pace 1, 2, 3; clamp at `X_MAX`; death at `X_MIN`; human diagonal movement (via `maze()` with scripted keys ending in ESC then `n`).
- Golden traces in `tests/fixtures/golden/*.json`, with a documented, explicit regeneration command. Regenerate only on intentional behaviour changes.

Acceptance: goldens pass against current `main`; results for D5 are written down in the PR description.

### PR 3 — Extract `Simulation.step()` (item 4)

- Add `core/simulation.py` per section 3.1, pace still read from `Game` (time logic untouched in this PR).
- Rewire `maze()` and `controlled_run()` onto `Simulation`; delete the duplicated blocks. Preserve the `controlled_run` callback payload (`action`, `score_increased`, `closest_line`) and the "keyboard overrides the wrapper action" behaviour.
- Apply D4 (single reset path).

Tests:

- Golden traces from PR 2 must pass **unchanged** (except the D4 restart case, regenerated with a note).
- `tests/unit/test_simulation.py`: per-direction blocking, `StepResult` fields (`score_delta` honours the floor, `moved` is the actual displacement), termination.
- Existing `tests/integration/test_game_engine_integration.py` still passes.

Acceptance: `engine.py` shrinks substantially; no behaviour change.

### PR 4 — Simulated time and episode caps (items 3, 5)

- Add config constants (section 3.3). Move pace progression into `Simulation.step()`; remove it from `Game.updateScreen()`.
- `start_pace` and `max_steps` plumbed through `Simulation`; add `--start-pace` CLI flag. `max_steps` is never set by human play.
- Update tests that assert pace via wall-clock seconds. Known candidates (line numbers approximate): `tests/unit/test_game.py` (~323-337), `tests/functional/test_gameplay_scenarios.py` (~211), `tests/integration/test_game_engine_integration.py` (~298), `tests/performance/test_performance_benchmarks.py` (~372). Derive tick counts from `config.PACE_UPDATE_TICKS`; no hard-coded frame counts.

Tests:

- Determinism: run 2 `Simulation`s with the same seed and intents, assert identical state after 5000 steps (no sleeps, no clock mocks needed).
- Pace increments at exactly `PACE_UPDATE_TICKS`; `start_pace` honoured; `truncated` at `max_steps`; a stalled agent terminates via truncation.
- Speed sanity (marked `performance`): 10k steps run without delay.

Acceptance: simulation tests run without any mocked clock; human play feel matches within D2.

### PR 5 — Record / replay (item 6)

- `core/replay.py`, `--record` / `--replay` / `--headless`, `.gitignore` entry for `replays/`.
- Record hook in the human loop (`maze()`), writing on episode end or quit.
- Commit 2-3 small regression replays in `tests/fixtures/replays/` with `expected` states.

Tests:

- Round trip: record a scripted session, replay headless, `expected` matches.
- RLE encode/decode edge cases (empty, single step, long runs).
- Version/maze-config mismatch gives a clear error instead of silent divergence.
- A tampered `expected` makes `--headless` exit non-zero.

Acceptance: a bug can be reported as one JSON file and reproduced headless.

## 6. Documentation to update (with the PR that changes the behaviour)

- [docs/architecture.md](../docs/architecture.md): new `Simulation` layer, tick-based pace.
- [docs/api-reference.md](../docs/api-reference.md): `Simulation`, `StepResult`, `generateMaze(rng=)`, `maze(seed=)`, `controlled_run(..., seed=)`, replay format.
- [docs/player-guide.md](../docs/player-guide.md) and [README.md](../README.md): CLI flags, seed display, pace interval note.
- [docs/testing-guide.md](../docs/testing-guide.md): determinism tests, golden trace regeneration, replay fixtures.

## 7. Risks and open questions

| Risk / question | Mitigation / decision needed |
|---|---|
| Golden traces are brittle if regenerated carelessly | Regeneration is an explicit documented command; the PR must explain any golden diff |
| Hidden dependence on the `random` global elsewhere (outside the repo, for example your earlier ML wrappers) | Search your training code for `random.seed`; after PR 1 seeding goes through `seed=` only |
| D2 pace timing change | Confirm acceptable, or switch to a constant derived from `FPS_DELAY_MS` |
| Collision tunnelling at pace >= 2 (D5) | Decide after PR 2 whether to fix and in which PR |
| `Game` still calls `pygame.init()` headless | Accepted limitation here; revisit only if profiling shows it matters for training throughput |
| Python 3.8 compatibility | No `X | None`, no walrus-only tricks, use `typing` generics; `dataclasses` is fine |
| Does restart with explicit seed (D1) surprise you? | Confirm preferred behaviour |

## 8. Definition of done (Plan 1)

- `infinite-maze --seed N` is reproducible; seed visible in HUD and log.
- Golden traces prove no unintended gameplay change.
- One rules implementation (`Simulation.step`) used by human play, `controlled_run`, replay and tests.
- Pace and scroll depend on ticks only; simulation runs with no sleeping and no wall clock.
- Replay files reproduce games headless and are used as regression fixtures.
- Maze generator verified (connected spanning tree, solvable) across many seeds and sizes.
- CI parity commands pass; docs updated.
