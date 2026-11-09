# Plan 2 — ML Debug Aids (dev branch only)

Status: proposed
Scope: observation and reward visualisation overlay, baseline policies with a fixed-seed evaluation harness.
Branch: a long-lived `ml-training` dev branch, **not** merged to `main` until the ML work is ready.
Prerequisite: [Plan 1](./01-foundations-and-debug-tooling.md) merged to `main` (`Simulation.step`, seeds, simulated time, replay, maze verification).

## 1. Goals and non-goals

**Goals**

- Make the agent's view of the world and its reward signal *visible* and *testable* before any learning algorithm is chosen.
- Provide baselines (random, simple heuristic) and a fixed-seed evaluation harness, so any later trained policy has a yardstick.
- Keep the game package clean: ML-specific code lives outside `infinite_maze/`.

**Non-goals**

- No training loop, algorithm choice, Gymnasium environment or neural network (later phases).
- No ML dependencies. Everything in this plan is stdlib + `pygame`, which keeps the branch runnable with no extras. NumPy can arrive with the Gymnasium wrapper.
- No changes to game rules. If a rules bug shows up, fix it on `main` and rebase.

## 2. Branch and layout strategy

```text
main  ----o----o----o----------o----->   (Plan 1 PRs, then ongoing fixes)
           \                    \
ml-training  o--o--o--o--o--o----o--o->  (rebased on main regularly)
```

- Rebase `ml-training` onto `main` after every Plan 1 PR; do not let it drift.
- Code lives in a new top-level package, outside the game package:

```text
training/
  __init__.py
  config.py        # observation/reward constants, seed sets, eval settings
  observation.py   # build_observation(sim) -> Observation
  reward.py        # RewardFunction, RewardBreakdown
  policies.py      # Policy base class, RandomPolicy, AlwaysRightPolicy, HeuristicPolicy
  evaluate.py      # evaluate(policy, seeds, ...) -> EvalReport, CLI
  overlay.py       # pygame overlay drawn on top of the rendered game
  runs/            # gitignored: JSON reports, logs
tests/unit/test_training_*.py
tests/functional/test_training_baselines.py
```

- Add `training/runs/` to `.gitignore`.
- CI currently lints `infinite_maze/` only. On this branch, extend black/flake8/mypy to `training/` so the new code is held to the same standard.
- Keep Python 3.8-compatible syntax unless the minimum version is deliberately raised later.

## 3. Shared definitions (needed by both items)

The overlay (item 7) and baselines (item 8) both depend on a concrete observation and reward, so these are defined first, as the first commits on the branch. They are small, pure functions of a `Simulation` and its last `StepResult`.

### 3.1 Observation (`training/observation.py`)

Recommended: egocentric wall raster plus scalars (cheaper and easier to debug than pixels).

```python
@dataclass
class Observation:
    grid: List[List[int]]      # GRID_SIZE x GRID_SIZE, 1 = wall in that raster cell
    rays: Tuple[float, float, float, float]  # normalised distance to nearest wall: right, left, up, down
    scalars: Tuple[float, ...] # x relative to [X_MIN, X_MAX], y relative to [Y_MIN, Y_MAX], pace, ticks fraction
```

- The raster is centred on the player. Every wall segment within the window marks each raster cell it crosses. This is independent of line scroll offsets (lines shift by `pace` pixels, so they do not stay aligned to cell boundaries).
- Constants in `training/config.py`: `OBS_GRID_SIZE` (default 9), `OBS_GRID_CELL_PX` (default 11, so the window is ~99 px, about 4.5 maze cells), `OBS_RAY_CAP_PX` (default 110).
- Cost control: filter lines by x-range before rasterising so the cost is not O(all lines).
- **Spike before committing:** compare raster vs ray-only observations on the heuristic baseline from item 8. If rays alone give the heuristic equal results, the raster may be unnecessary for a first run.

### 3.2 Reward (`training/reward.py`)

Kept separate from the in-game score (which is floored at 0 and is only +/-1).

```python
@dataclass
class RewardConfig:
    progress_weight: float = 1.0     # per pixel actually moved right (unfloored; left is negative)
    step_penalty: float = 0.01       # per step
    death_penalty: float = 50.0      # on terminated
    truncation_bonus: float = 0.0    # on truncated (survived the cap)

@dataclass
class RewardBreakdown:
    progress: float
    step: float
    terminal: float
    total: float
```

- Progress uses `StepResult.moved[0]` from Plan 1, so it is not subject to the score floor or to scrolling.
- Breakdown is returned (not just a scalar) so the overlay can show where reward comes from.
- Reward-hacking guards, each with a test:
  - left/right oscillation nets zero progress and a negative step penalty, so it must score below an always-right policy;
  - wall-hugging (no movement) accumulates only step penalty;
  - the score-floor exploit is impossible because the reward does not use the floored score;
  - dying is never better than surviving: `death_penalty` must exceed the step penalty accrued over any plausible remaining episode; assert this relationship for the configured `max_steps`.

## 4. Item 7 — Observation and reward overlay

**Purpose:** see exactly what the agent sees and is rewarded for, frame by frame, on a replayed or human-driven game.

### Design

- Smallest possible hook in the game package: `Game.setOverlay(callback: Optional[Callable[[pygame.Surface], None]])`. `Game.updateScreen()` calls it after drawing the maze and before `pygame.display.flip()`. No ML imports in `infinite_maze/`; the callback is supplied from `training/overlay.py`.
  - This is the only change to `infinite_maze/` on the dev branch. It is tiny and can be cherry-picked to `main` later if useful (a generic debug overlay is valid for any developer).
- `training/overlay.py` draws, toggled with `F3`:
  - the observation window: raster cells tinted where `grid == 1`, window border around the player;
  - the four rays with their lengths;
  - text panel: seed, tick, pace, last intent, step reward and its `RewardBreakdown`, cumulative return;
  - optional: player hitbox.
- Frame stepping while paused (`.` advances one `Simulation.step`). Stepping backwards is done by re-running from the start of the replay to step `N-1` (deterministic from Plan 1), not by snapshotting.
- Runner: `python -m training.overlay --replay path.json [--reward-config path]` loads a Plan 1 replay and plays it rendered with the overlay. The human-play variant `--seed N` runs `maze()` with the overlay enabled.
- Note: the seed in the HUD already ships with Plan 1, so the overlay does not need to duplicate it outside the panel.

### Tests

- Unit: `build_observation` on hand-built `Line` lists (empty maze, wall directly right, wall directly above, wall straddling the window edge); ray distances capped at `OBS_RAY_CAP_PX`; raster marks every cell a wall segment crosses.
- Unit: reward breakdown for each guard in section 3.2.
- Integration (headless, `PygameMockContext`): the overlay callback is invoked once per `updateScreen`, draws without raising, and does nothing when disabled.
- Invariance: attaching an overlay must not change `Simulation` state (run the same replay with and without it, assert identical `expected`).

### Acceptance

- Replaying a recorded session with `F3` shows the window and reward panel; values match the unit-tested functions.

## 5. Item 8 — Baselines and evaluation harness

**Purpose:** a yardstick for "the model learned something", plus a fuzz test that the game cannot crash under arbitrary input.

### Design

```python
class Policy(ABC):
    def reset(self, seed: int) -> None: ...
    @abstractmethod
    def act(self, obs: Observation) -> int: ...      # discrete action 0..4, same space as config.MOVEMENT_CONSTANTS
```

(An abstract base class is used instead of `typing.Protocol` so that the contract is explicit and a missing `act` fails at instantiation.)

Policies in `training/policies.py`:

- `RandomPolicy(rng)`: uniform over the 5 actions. Seeded for reproducibility.
- `AlwaysRightPolicy`: lower bound for "do nothing clever".
- `HeuristicPolicy`: move right if the right ray is open; otherwise move up or down toward the side with the longer ray, with hysteresis (commit to a vertical direction for N steps) to avoid oscillation. This is a simple reactive baseline, not a maze solver. Expect it to get stuck in dead ends; that is the point.

Evaluation in `training/evaluate.py`:

```python
def evaluate(policy, seeds: Sequence[int], max_steps: int, start_pace: int,
             reward: RewardConfig) -> EvalReport
```

- Runs the policy on `Simulation` directly (no rendering, no sleeping) for each seed.
- `EvalReport`: per-seed `{seed, score, steps, return, terminated, truncated}` plus mean/median/min/max and death rate.
- Fixed seed sets in `training/config.py`, defined once and never changed silently:
  - `TRAIN_SEEDS = range(0, 100)`
  - `VALIDATION_SEEDS = range(1000, 1050)`
  - `TEST_SEEDS = range(2000, 2050)`
- CLI: `python -m training.evaluate --policy heuristic --seeds validation --max-steps 5000 --out training/runs/heuristic.json`.
- Optional: write the worst-seed run as a Plan 1 replay file so it can be watched with the overlay.

### Tests

- Unit: policies return valid actions; `RandomPolicy` is reproducible given a seed; heuristic hysteresis does not flip every step in a symmetric corridor.
- Functional (`tests/functional/test_training_baselines.py`): `evaluate()` on 3 seeds with a small `max_steps` is deterministic across two runs (identical `EvalReport`).
- Fuzz (mark `slow`): `RandomPolicy` over 50 seeds, 2000 steps each, never raises and always ends `terminated` or `truncated`.
- Ordering sanity (mark `slow`): on `VALIDATION_SEEDS`, mean return of `HeuristicPolicy` > `AlwaysRightPolicy` > `RandomPolicy`. If this ordering does not hold, the reward or observation is suspect, which is exactly the signal this item exists to give.

### Acceptance

- One command produces a comparable table for all baselines on the fixed validation seeds, and results are reproducible run to run.
- Baseline numbers are recorded (in the PR description or `training/README.md`) as the bar a learned policy must clear.

## 6. Delivery order on the dev branch

1. Scaffold `training/` package, `.gitignore`, lint/type-check wiring for `training/`.
2. `observation.py` + `reward.py` + their unit tests (section 3). Include the raster-vs-rays spike result.
3. `policies.py` + `evaluate.py` + tests (item 8). This gives numbers early.
4. `Game.setOverlay` hook + `overlay.py` + tests (item 7).
5. Calibrate: look at baseline replays with the overlay; adjust `RewardConfig` and observation constants; re-run baselines.

Item 8 comes before 7 in time even though the plan numbering is reversed, because baselines validate the observation and reward numerically and the overlay is then used to explain surprises.

## 7. Risks and open questions

| Risk / question | Mitigation / decision needed |
|---|---|
| Observation window too small to see gaps ahead at high pace | Make `OBS_GRID_SIZE` / `OBS_GRID_CELL_PX` configurable; evaluate heuristic with larger windows |
| Reward shaping may encourage exploits not covered by the guards | Keep the breakdown visible; add a regression test per exploit found |
| Branch drift from `main` | Rebase after each Plan 1 PR; keep the branch's footprint in `infinite_maze/` to the single overlay hook |
| Python minimum version when ML libs arrive | Decide separately (repo declares `^3.8.1`); put ML libs in an optional Poetry group, never in core dependencies |
| Seed sets chosen here become the permanent benchmark | Agree on `TRAIN/VALIDATION/TEST` ranges before the first real training run; do not reuse test seeds for tuning |
| Should the overlay hook be upstreamed to `main`? | Optional; it is generic. Decide when Plan 2 stabilises |

## 8. Definition of done (Plan 2)

- `build_observation` and `RewardFunction` are pure, tested, and configurable from `training/config.py`.
- Overlay shows observation window, rays, step reward breakdown and cumulative return on any replay.
- Random, always-right and heuristic baselines produce reproducible reports on the fixed validation seeds, with the expected ordering.
- `training/` passes black/flake8/mypy; `infinite_maze/` differs from `main` only by the overlay hook.
- No new core dependencies were added.
