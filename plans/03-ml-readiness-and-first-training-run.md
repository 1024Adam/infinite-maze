# Plan 3 — ML Readiness and First Training Run

Status: proposed
Scope: measure and fix throughput, build an oracle (privileged solver), build a Gymnasium environment, add the optional ML dependency groups, build the training/evaluation harness, and run the first fixed-seed experiment.
Prerequisites: [Plan 1](./01-foundations-and-debug-tooling.md) and [Plan 2](./02-ml-debug-aids.md) implemented (they are).
Outcome of the readiness review: the repo is deterministic and headless-steppable with baselines, but has no environment, unmeasured (probably poor) throughput, and no ceiling to compare against.

## 1. Decisions adopted

| # | Decision | Choice | Effect on this plan |
|---|---|---|---|
| D1 | Algorithm family | **PPO** (Stable-Baselines3) with **action repeat** (`frame_skip`) | Environment must support vectorised subprocess copies and `frame_skip`. DQN or a recurrent PPO can be revisited later. |
| D2 | Python minimum for ML | Core stays `^3.8.1`; ML libraries in **optional Poetry groups** with `python = ">=3.10"` | Core game and its CI are unaffected. ML tests skip when the groups are not installed. |
| D3 | "Expert" metric | **Oracle-fraction ladder**, reported separately for the training seed and held-out `validation` seeds. Kind chosen now; numeric thresholds set after WP3 shows what is achievable | `training/expert.py`, thresholds in `training/config.py`. |
| D4 | Oracle-based reward shaping | **Deferred.** Run plain reward first; adopt only if the plain run stalls | Not in this plan's critical path. Section 9 records the trigger and sketch. |

## 2. Findings this plan responds to

1. The baselines stall early (heuristic mean score ≈ 101 over 2000 steps, about 4.5 cells). There is no ceiling to measure against.
2. `Observation` is a dataclass of nested lists; there is no Gymnasium environment, no spaces, no numpy conversion, no training script.
3. Throughput is unmeasured. From reading the code (not measured):
   - `Line.generateMaze` rebuilds a `sets` list with list-membership checks after every wall removal;
   - `Simulation.reset()` regenerates the maze even for an unchanged seed;
   - every step scans all ~1,100 lines for collision and for the observation, and, when the player is pinned at `X_MAX`, mutates every line to scroll;
   - [the throughput test](../tests/performance/test_performance_benchmarks.py) allows 60 s for 10k steps, so it cannot catch regressions.
4. `ticksFraction` in [observation.py](../training/observation.py) changes meaning with `max_steps` (steps/cap versus a pace-interval sawtooth), so train and eval observations can differ.
5. `RewardConfig.validate` requires `death_penalty > step_penalty * max_steps`; with the default step penalty of 0.01, long horizons need an explicit reward config.
6. Episodes of 2000 steps never reach meaningful scroll pressure (pace 1 scrolls 1 px per 10 ticks and pace rises every 1800 ticks), so "expert" play needs long horizons or a `start_pace` curriculum.
7. A perfect maze cannot be solved from a 4.5-cell window without memory. Expect a fixed-seed run to be able to memorise; held-out seeds will need history (frame stacking or a recurrent policy). This is a known limit of the first run, not a bug.

## 3. Ownership and branches

| Work | Owner agent | Branch / merge target |
|---|---|---|
| Anything under `infinite_maze/` (performance, `Simulation` API additions) | `pygame-dev` | Small PRs to the default branch (`master`), because they benefit everyone |
| Anything under `training/`, `pyproject.toml` ML groups, ML tests, experiment configs | `ml-dev` (see section 11) | The long-lived ML dev branch |

- [tests.yml](../.github/workflows/tests.yml) triggers on pushes to `master` and `dev/**`. Name the ML branch `dev/ml-training` (or similar) so CI runs on push. A branch called just `ml-training` gets no CI until a PR is opened.
- Rebase the ML branch on the default branch after each `infinite_maze/` PR.
- The ML agent never edits `infinite_maze/`; it files a scoped request for `pygame-dev` instead.

## 4. Work packages

```text
WP1 benchmark ──► WP2 perf (pygame-dev, master)
      │                    │
      ├──► WP3 oracle ◄────┘ (needs WP2a/b only for speed, not correctness)
      │         │
      │         └──► WP7 expert ladder + ceilings
      └──► WP4 deps ──► WP5 env ──► WP6 train/eval ──► WP8 first run
```

WP1, WP3 and WP4 can start in parallel. WP2 is gated by WP1's numbers. WP8 needs everything.

### WP1 — Benchmark (ML branch) — first

New `training/benchmark.py`: `python -m training.benchmark [--seeds N] [--steps N] [--profile]`.

Measures, per seed and in aggregate (median and p95):

- `Line.generateMaze` time and `Simulation.reset()` time (with the same seed repeated, and with new seeds);
- `Simulation.step` steps/s with a scripted fuzz policy, with `build_observation` included and excluded;
- `--profile`: `cProfile` top 15 for a 10k-step run, so the real hot spots are ranked (scroll mutation, collision scan, observation scan, or something else).

Write the baseline numbers into `training/README.md` (machine, Python version, date).

Acceptance: one command prints a table and a ranked hot-spot list; numbers are committed to the README. **WP2's scope and targets are set from this output.**

Provisional targets (to confirm against the baseline): reset ≤ 100 ms; ≥ 5,000 steps/s single core including observation; a 10× improvement over the measured baseline if the baseline is below 1,000 steps/s.

### WP2 — Performance fixes (`pygame-dev`, to the default branch)

Only the items WP1 shows to matter. All are guarded by one rule: **identical behaviour**. [Golden traces](../tests/fixtures/golden) and replays must pass unchanged, and seeded mazes must not change (otherwise committed replays, seed sets and baseline numbers become invalid).

**WP2a — Faster `generateMaze`.**
- Replace the rescan-per-iteration with incremental set tracking (union-find or label→lines map plus a count of distinct labels still present on remaining lines).
- Preserve the exact RNG draw sequence (one `rng.randint(0, len(lines) - 1)` per iteration, same list order after deletions), the exact termination condition, and the final `sideA`/`sideB` labels.
- Test: freeze the current algorithm as a reference implementation in `tests/fixtures/legacy_maze.py` and assert identical line lists (coordinates and side labels) for seeds 0-199 across a few `(width, height)` pairs. Delete the reference only if you later accept changing all seeds.

**WP2b — Faster per-step work.** Choose using the profile. Candidate approaches, in order of preference:
1. *Column index for collision and observation scans.* Lines in the same column share `xStart`; only the few columns near the player matter.
2. *Lazy scroll.* If scroll mutation dominates (it runs on nearly every step once the player is pinned at `X_MAX`), keep an accumulated scroll offset and materialise line coordinates only when something needs them (rendering, overlay, golden hash, replay state), via `Simulation.getLines()`.
3. Both.

Constraints: public `Line` getters and `Simulation.getLines()` keep returning screen-space coordinates; getter/setter conventions retained; Python 3.8 compatible.

**WP2c — Reset cache (only if reset is still above target).** Cache the pristine maze per seed (small LRU of coordinate tuples) and rebuild `Line` objects from it on reset.

**WP2d — Expose scroll for the oracle.** Add `Simulation.getScrollOffset()` returning total pixels the world has scrolled since reset (including recycling shifts), so external code can convert screen coordinates into world and tile coordinates. Additive, with a unit test.

**WP2e — Tighten the throughput test.** After the numbers are known, set the `TestSimulationThroughput` limit to about 3× the measured time, so regressions fail.

Acceptance: goldens and legacy-maze equivalence pass; WP1 re-run shows the targets met; CI parity commands pass.

### WP3 — Oracle policy (ML branch)

Purpose: the ceiling for the expert ladder, a sanity check that the maze is solvable under the real physics, and a source of demonstrations.

New `training/oracle.py`:
- `PrivilegedPolicy(Policy)` with `act_privileged(sim) -> int` (it may read walls; it is not an observation-based policy). `evaluate()` gains support for privileged policies.
- Build the tile's cell graph once per seed from freshly generated lines using [maze_analysis.buildCellGraph](../infinite_maze/entities/maze_analysis.py). The world is periodic with 30 columns and the wrap between the last and first column is open in every row, so path search runs on a periodic graph.
- World-to-cell mapping uses `Simulation.getScrollOffset()` (WP2d) and the player's position; BFS target is a column several cells ahead of the current one (so it keeps moving as the world scrolls).
- Step control converts "go to the next cell" into one of the five discrete actions at pixel level: centre the 10 px player in the 22 px corridor before crossing, honour the 1 px walls, avoid the left kill boundary.
- Do **not** modify game rules to make the oracle succeed. If it cannot solve a seed under the real collision rules, record that as a finding.

New `training/oracle_stats.py` (`python -m training.oracle_stats`): for `start_pace` 0..N and fixed seeds, report mean score at a step cap, steps to clear the first tile versus the shortest-path lower bound, pace at death and highest sustainable pace, and death rate. These ceilings choose the `start_pace` curriculum and the expert-ladder thresholds (WP7).

Demonstrations: `--export-demos path` writes `(flattened observation, action)` pairs for later behaviour cloning. The export format is a plain JSONL file, no numpy.

Tests: a hand-built tiny maze (known path) solved in the expected number of steps; oracle never enters a wall (reuse `_overlapsAnyWall` logic); oracle reproducible; oracle solves the first tile on 20 seeds at pace 0.

Acceptance: oracle solves the first tile on all `train` seeds 0-99 at pace 0 (or each failure is explained), and `oracle_stats` produces the ceiling table.

### WP4 — Optional ML dependencies (ML branch)

Edit [pyproject.toml](../pyproject.toml). Use two optional groups so CI can install the light one without PyTorch:

```toml
[tool.poetry.group.ml-env]
optional = true
[tool.poetry.group.ml-env.dependencies]
gymnasium = {version = "...", python = ">=3.10"}
numpy     = {version = "...", python = ">=3.10"}

[tool.poetry.group.ml-train]
optional = true
[tool.poetry.group.ml-train.dependencies]
stable-baselines3 = {version = "...", python = ">=3.10"}
torch             = {version = "...", python = ">=3.10"}   # CPU build
```

- Versions are chosen at implementation time (latest compatible; resolve with `poetry lock` if the repo keeps a lock file). `...` above is deliberate.
- Core `[tool.poetry.dependencies]` is untouched; `poetry install` without groups still yields a working game.
- Add mypy overrides (`ignore_missing_imports`) for `gymnasium.*`, `numpy.*`, `stable_baselines3.*`, `torch.*`, since the CI type-check does not install these groups.
- CI: add a step or job that runs `poetry install --with ml-env` and the env tests. Do **not** install `ml-train` (PyTorch) in CI.
- PyTorch CPU wheels can be awkward with Poetry; document the working install command in `training/README.md`.

Acceptance: `poetry install` (no groups) passes the existing CI commands; `poetry install --with ml-env,ml-train` succeeds on Python ≥ 3.10; tests that need these libraries use `pytest.importorskip`.

### WP5 — Gymnasium environment (ML branch)

First, small change to the observation (stdlib only) in `training/observation.py`:
- Make the last scalar always `(ticks % PACE_UPDATE_TICKS) / PACE_UPDATE_TICKS` (rename to `pace_phase`), independent of `max_steps`. Update the tests and `training/README.md`.
- Add `flatten_observation(obs) -> List[float]` (grid row-major, rays, scalars), the single flattening used by the environment, `ModelPolicy` and any demonstration export.

New `training/env.py`:

```python
class MazeEnv(gymnasium.Env):
    def __init__(self, seeds: Sequence[int], start_pace: int = 0,
                 max_steps: int = ..., frame_skip: int = 4,
                 reward_config: Optional[RewardConfig] = None) -> None: ...
    action_space = Discrete(5)                    # config.MOVEMENT_CONSTANTS
    observation_space = Box(0.0, 1.0, shape=(flat_size,), dtype=np.float32)
    def reset(self, *, seed=None, options=None): ...   # (obs, info)
    def step(self, action): ...                        # (obs, reward, terminated, truncated, info)
```

- Maze seed handling: `reset(seed=s)` seeds Gymnasium's `np_random`; the **maze seed** is drawn from `seeds` with that RNG (a single-element list gives the fixed-seed experiment), and it is always passed to `Simulation.reset(seed=...)` explicitly (the simulation otherwise reuses the last explicit seed). `info["maze_seed"]` reports it.
- `frame_skip`: repeat the chosen action for `frame_skip` ticks, summing rewards, stopping early on termination or truncation.
- `max_steps` is in simulation ticks and passed to `Simulation`, so truncation semantics are identical everywhere.
- `RewardConfig.validate(max_steps)` runs at construction. For the first run pick explicit values (for example `step_penalty=0.001`, `death_penalty=100`) in the experiment config rather than changing the defaults.
- `info` includes score, ticks, pace, `maze_seed`.
- SDL: set `SDL_VIDEODRIVER`/`SDL_AUDIODRIVER` to `dummy` with `os.environ.setdefault` before pygame is imported in the environment module.

New `training/model_policy.py`: `ModelPolicy(Policy)` wraps an SB3 model, using `flatten_observation` and deterministic actions. `evaluate()` gains `frame_skip` so a model is evaluated with the same action repeat it was trained with.

Tests (`tests/unit/test_training_env.py`, skipped without `ml-env`):
- `gymnasium.utils.env_checker.check_env` passes.
- Same seeds and same action sequence give identical observations and rewards across two environments.
- Random-agent smoke test over 20 seeds never raises and always ends terminated or truncated.
- `frame_skip=k` gives the same final state as k manual `Simulation.step` calls.
- Reward sum over an episode equals the harness's `total_return` for the same actions.

Acceptance: a random agent runs 100 episodes through the environment; environment steps/s are recorded in the README.

### WP6 — Training and evaluation harness (ML branch)

New `training/train.py`: `python -m training.train --config training/configs/<name>.json [--resume DIR]`.

Config (JSON, committed; the run copies it into its output directory so runs are pinned):

```json
{
  "name": "v1_fixed_seed0",
  "algo": "ppo",
  "train_seeds": [0],
  "eval_seeds": {"train": [0], "validation": "validation"},
  "start_pace": 4,
  "curriculum": [],
  "max_steps": 5000,
  "frame_skip": 4,
  "reward": {"progress_weight": 1.0, "step_penalty": 0.001, "death_penalty": 100.0, "truncation_bonus": 0.0},
  "total_timesteps": 2000000,
  "n_envs": 8,
  "ppo": {"policy": "MlpPolicy", "net_arch": [64, 64], "n_steps": 512, "batch_size": 256,
          "learning_rate": 0.0003, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.01},
  "eval_every_timesteps": 100000,
  "seed": 0
}
```

The numbers above are starting points, not tuned values; pick `start_pace` from the WP3 ceiling table, and `n_envs` from the machine.

Per run, written to `training/runs/<name>/` (gitignored): `config.json` (copy), `metadata.json` (git commit, library versions, Python, platform, seed), `metrics.csv` (timesteps, mean return, mean score, episode length, losses), `checkpoints/`, `best.zip`, `eval.json`, and `replays/` (best and worst eval episodes as [Plan 1 replays](../tests/fixtures/replays) so they can be watched with the overlay).

Evaluation callback (every `eval_every_timesteps`): run the shared `evaluate()` harness through `ModelPolicy` on (a) the training seed(s) and (b) the `validation` seeds, with deterministic actions, and log both. Keeping them separate is what distinguishes memorisation from generalisation.

Seeding: numpy, torch, SB3 and environment seeds all derive from `config.seed`; identical config and seed on the same machine should reproduce the learning curve (note any known nondeterminism, for example multi-process ordering).

Curriculum (optional, off by default): `curriculum` is a list of `{"timesteps": N, "start_pace": p}` stages, applied by recreating the training environments.

`python -m training.evaluate` gains `--policy model --model path --frame-skip k`, and `--policy oracle`.

Tests: config parsing and validation (unknown keys rejected), run-directory layout, a 2,000-timestep smoke run (marked `slow`, skipped without `ml-train`) that writes all expected artefacts.

Acceptance: a smoke run completes in minutes and produces every artefact; the final model can be evaluated with `training.evaluate` and replayed with `training.overlay`.

### WP7 — Expert ladder and ceilings (ML branch)

New `training/expert.py`: `ladder(model_report, heuristic_report, oracle_report) -> LadderResult`.

| Rung | Criterion (all measured on the same seeds, `start_pace` and `max_steps`) |
|---|---|
| 1 | Mean return above the heuristic baseline |
| 2 | Mean score ≥ R2 × oracle mean score |
| 3 | Mean score ≥ R3 × oracle mean score |
| 4 | Survives to pace ≥ P with death rate ≤ X |

- Thresholds `R2`, `R3`, `P`, `X` live in `training/config.py` as `EXPERT_LADDER`; **their initial values are set after WP3**, not now (suggested starting point: R2 = 0.5, R3 = 0.9).
- Every ladder result is reported twice: on the training seed(s) and on the validation seeds.
- A fixed-seed result is always labelled as potentially memorised.
- Re-run the heuristic and oracle baselines with the final settings, so every comparison is like for like. Record them in `training/README.md`.

### WP8 — First experiment (ML branch)

`training/configs/v1_fixed_seed0.json` as above. Procedure:

1. Re-run baselines (random, right, heuristic, oracle) at the chosen `start_pace` and `max_steps`; record them.
2. Short smoke run, then the full run, with the config committed before launch.
3. Compare learning curve and ladder results; look at the best and worst replays in the overlay.
4. Write findings (what stalled, where) into `training/README.md` or a run report.

Expected outcome, stated so it can be judged: the fixed-seed agent should clear rung 1 quickly; whether it reaches rungs 2-3 shows whether plain reward is enough. If it stalls in the first dead end, trigger the shaping work in section 9. Do not read held-out performance as a failure of the pipeline; it is the next experiment.

## 5. Delivery order and PR sizing

1. WP1 benchmark (ML branch).
2. WP4 dependencies and the `ticksFraction` / `flatten_observation` change (ML branch).
3. WP2a/b/c/d/e (`pygame-dev`, default branch), sized to WP1's findings, one PR each.
4. WP3 oracle (ML branch), starting as soon as WP2d lands.
5. WP5 environment, then WP6 harness, then WP7 ladder (ML branch).
6. WP8 first run.

CI parity for any PR touching `infinite_maze/` or `training/`:

```bash
poetry run black --check infinite_maze/ training/
poetry run flake8 infinite_maze/ training/
poetry run mypy infinite_maze/ training/
poetry run pytest -m "not slow and not performance"
```

## 6. Documentation to update

- [training/README.md](../training/README.md): benchmark numbers, oracle ceilings, environment, training and evaluation commands, install command for the ML groups, ladder definition and results.
- [docs/api-reference.md](../docs/api-reference.md) and [docs/architecture.md](../docs/architecture.md): `Simulation.getScrollOffset()` and any performance-related internals changed in WP2.
- [docs/testing-guide.md](../docs/testing-guide.md): optional-dependency tests and how they skip; the legacy-maze equivalence test.

## 7. Risks and open questions

| Risk / question | Mitigation |
|---|---|
| WP2 changes alter maze output or physics | Legacy-maze equivalence test plus unchanged golden traces; any intentional change is its own PR with regenerated goldens and baselines |
| Oracle cannot solve some seeds under real collision rules | Report as a finding; do not alter the rules to rescue it |
| Oracle's pixel-level control is fiddly (10 px player in 22 px corridors) | Hand-built tiny-maze tests first; allow it to be slower than optimal, it only needs to be a good ceiling |
| PyTorch install friction with Poetry | Separate `ml-train` group; CPU build only; documented install command |
| Parallel environments plus pygame init | `SDL_*` dummy drivers set in the environment module; smoke test with `SubprocVecEnv` in WP6 |
| Reward horizon constraint (`validate`) | Explicit reward values in experiment configs; reconsider whether the check should stay a hard error once real runs exist |
| Fixed-seed success mistaken for generalisation | Report train and validation separately; label fixed-seed results |
| Long runs are expensive and not reproducible if configs change mid-run | Config committed before launch and copied into the run directory; long runs launched by the user |
| Open question | Thresholds `R2`, `R3`, `P`, `X`: set after WP3 |
| Open question | Whether to later add frame stacking or recurrent PPO (needed for held-out seeds) |

## 8. Definition of done (Plan 3)

- Benchmark numbers committed; targets met after WP2 (or the shortfall is documented with a profile).
- Seeded mazes, goldens and replays unchanged by the performance work.
- Oracle solves the first tile on all `train` seeds (or failures explained) and the ceiling table exists.
- `MazeEnv` passes `check_env`, is deterministic for a given seed and action sequence, and runs under vectorised subprocess copies.
- Core `poetry install` and CI are unaffected; ML libraries live only in optional groups.
- One fixed-seed PPO run completed from a committed config, with metrics, checkpoints, evaluation on train and validation seeds, replays, and a ladder report.

## 9. Deferred: oracle-based reward shaping (decision D4)

Trigger: the plain-reward fixed-seed run stalls at the first dead end (which the heuristic baseline already does).

Sketch: potential-based shaping `r' = r + γ·Φ(s') − Φ(s)` with `Φ(s) = −(path distance from the player's cell to a target column ahead)`, computed from the same oracle machinery. It leaves the optimal policy unchanged in theory, is used only during training, and evaluation still uses the real score. Caveats to settle then: the potential must stay consistent as the world scrolls and recycles (use world coordinates via `getScrollOffset()`), and shaping mainly helps memorisation on fixed seeds, not generalisation to unseen mazes. Alternative or complement: behaviour cloning from the WP3 demonstrations, then RL fine-tuning.

## 10. Out of scope

- Recurrent or frame-stacked policies and training on many seeds (the next experiment after the fixed-seed run).
- DQN and other algorithms.
- Pixel observations and CNNs.
- Changes to game rules or the collision model.

## 11. Agent roles

- `pygame-dev` keeps `infinite_maze/` (WP2). Its constraints ("no dependencies beyond pygame", CI commands covering `infinite_maze/` only) are correct for that scope and would conflict with ML work.
- A new **`ml-dev`** agent owns `training/`, the ML dependency groups, ML tests and experiment configs. Suggested rules: never edit `infinite_maze/` (file a request for `pygame-dev`); every experiment starts from a committed config; smoke runs (a few minutes) may be run by the agent, long runs are launched by the user; report results in the standard ladder and baseline tables; keep core dependencies untouched; Python 3.8-compatible syntax in stdlib-only modules, `>=3.10` only inside modules that need the ML groups.
- Running experiments is handled by the same agent plus a reusable skill (`.github/skills/run-experiment/`: pin config, smoke run, launch command, report template), rather than a third agent.
