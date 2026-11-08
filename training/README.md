# Training tools

ML-side tooling for Infinite Maze. It lives outside `infinite_maze/` so the game
stays free of ML code and dependencies (stdlib + `pygame` only).

| Module | Purpose |
|---|---|
| `config.py` | Observation/reward constants, **fixed seed sets**, evaluation and overlay settings |
| `observation.py` | `build_observation(sim)`: egocentric wall raster, four rays, scalars |
| `reward.py` | `RewardFunction` / `RewardBreakdown` / `RewardConfig` |
| `policies.py` | `RandomPolicy`, `AlwaysRightPolicy`, `HeuristicPolicy` (+ `RasterHeuristicPolicy`) |
| `evaluate.py` | `evaluate(policy, seeds, ...)`, comparison table CLI |
| `overlay.py` | Observation/reward overlay for replays and human play |
| `benchmark.py` | `python -m training.benchmark`: generation, reset and step throughput |
| `oracle.py`, `oracle_stats.py` | Privileged shortest-path solver and its ceiling table / demo export |
| `stepping.py` | `repeat_action`: the one action-repeat (`frame_skip`) implementation |
| `experiment.py` | Experiment config (JSON) parsing and validation, stdlib only |
| `env.py` | `MazeEnv`, the Gymnasium environment (needs `ml-env`) |
| `model_policy.py`, `train.py` | SB3 model as a `Policy`; PPO training harness (needs `ml-train`) |
| `expert.py` | Expert ladder (`ladder()` and run report CLI) |

## Install

The game and the stdlib-only training modules need only the core install. The ML
libraries are optional Poetry groups (Python >= 3.10):

```bash
poetry install --with ml-env              # Gymnasium + numpy: MazeEnv and its tests
poetry install --with ml-env,ml-train     # + Stable-Baselines3 + PyTorch: training
```

`poetry.lock` is gitignored in this repository, so there is nothing to re-lock. If
the default PyTorch wheel is too large, install the CPU build first:
`poetry run pip install torch --index-url https://download.pytorch.org/whl/cpu`.
CI installs only `ml-env`, never `ml-train`. Without the groups the ML tests skip
(`pytest.importorskip`).

## Observation

`build_observation(sim)` returns an `Observation`:

- `grid`: `OBS_GRID_SIZE x OBS_GRID_SIZE` raster (default 9 x 9 cells of 11 px), centred on
  the player; `1` where a wall crosses the cell. Independent of scroll offsets.
- `rays`: distance to the nearest wall right, left, up, down, as a fraction of
  `OBS_RAY_CAP_PX`. Walls touching the player's box count, like the engine's collision.
- `scalars`: x, y (within the play area), pace and `pace_phase`, all in `[0, 1]`.
  `pace_phase` is `(ticks % PACE_UPDATE_TICKS) / PACE_UPDATE_TICKS`, independent of
  `max_steps` (it used to be steps/cap when a cap was set, so train and eval differed).

`flatten_observation(obs)` returns `[grid row-major (81), rays (4), scalars (4)]`
(89 floats in `[0, 1]`). It is the single flattening used by `MazeEnv`,
`ModelPolicy` and the oracle demo export.

## Reward

Separate from the in-game score (which is floored at 0). Per step:
`progress_weight * moved_x - step_penalty`, minus `death_penalty` on termination and
plus `truncation_bonus` on truncation. `RewardConfig.validate(max_steps)` rejects a
death penalty that does not exceed the step penalty accrued over a whole episode, so
dying can never beat surviving. `RewardBreakdown` keeps the parts visible.

The default `death_penalty` is 100 (not 50) so the check also passes for 5000-step
episodes.

## Evaluating baselines

```bash
poetry run python -m training.evaluate --policy all --seeds validation \
    --max-steps 2000 --out training/runs/baselines.json
```

Options: `--policy {random,right,heuristic,heuristic-raster,all}`,
`--seeds {train,validation,test}`, `--num-seeds N`, `--start-pace N`,
`--reward-config path.json`, `--worst-replay-dir DIR` (writes each policy's
worst-seed episode as a replay you can watch with the overlay).

Seed sets are permanent: `train` 0-99, `validation` 1000-1049, `test` 2000-2049. Do not
tune on `test`.

### Baseline numbers (validation seeds, `--max-steps 2000`, default reward)

| Policy | Mean return | Median | Mean score | Mean steps | Death rate |
|---|---|---|---|---|---|
| random | -102.5 | -101.1 | 0.8 | 231 | 100% |
| right | 38.1 | 24.0 | 58.1 | 2000 | 0% |
| heuristic | 81.1 | 78.0 | 101.1 | 2000 | 0% |

A learned policy has to clear the heuristic's mean return of 81.1 on these seeds.
Episodes are capped at 2000 steps, which is just past the first pace increase (tick
1800), so death rate for the non-random policies is 0%; use `--start-pace` or a larger
`--max-steps` to stress them.

### Frame skip

`evaluate(..., frame_skip=k)` and `--frame-skip k` repeat each chosen action for `k`
ticks (`stepping.repeat_action`, shared with `MazeEnv`). `max_steps` stays in
simulation ticks and the episode stops mid-repeat at the cap. Evaluate a model with
the `frame_skip` it was trained with.

### Raster vs rays

The same heuristic reading only the raster (`heuristic-raster`) scores a mean return
of 63.7 on the same seeds versus 81.1 with rays. The raster's 11 px resolution cannot
see a wall that is 2 px away, so for a first learning run the rays alone are the
stronger signal; the raster is still available for a convolutional model.

## Overlay

```bash
poetry run python -m training.overlay --replay path.json [--reward-config r.json] [--hitbox]
poetry run python -m training.overlay --seed 1234        # play yourself
```

Draws the observation window (orange = wall cells), the four rays, and a panel with
seed, tick, pace, last intent, step reward breakdown and cumulative return.

- `F3`: toggle the overlay.
- Replay mode: `SPACE` pause, `.` step forward and `,` step back (while paused; going
  back re-runs the replay from its start), `ESC` quit.

The only change to `infinite_maze/` for this is `Game.setOverlay(callback)`.

## Benchmark

```bash
poetry run python -m training.benchmark --seeds 5 --steps 10000 [--profile]
```

Measured on 2026-11-08 (WSL2, 20 cores, Python 3.12.3, `--seeds 5 --steps 10000`,
medians):

| Measurement | Before | After (`generateMaze` + recycle changes) |
|---|---|---|
| `Line.generateMaze` | 778 ms | 5.1 ms |
| `Simulation.reset` (same seed / new seeds) | 789 / 782 ms | 6.2 / 5.5 ms |
| steps/s, step only | 6,900 | 14,500 |
| steps/s, with `build_observation` | 4,800 | 7,900 |

The profile before the changes was dominated by `generateMaze` (rebuilding the set
list after every wall removal), then `_isBlocked`, `build_observation` and
`_recycleLines`. Targets from the plan (reset <= 100 ms, >= 5,000 steps/s with
observation) are met. `MazeEnv` with `frame_skip=4`: about 2,000 environment steps/s
on 100 random episodes (these are short, so resets are a visible share).

Performance changes in `infinite_maze/` keep seeded mazes identical:
`tests/unit/test_maze_equivalence.py` compares `Line.generateMaze` with the frozen
original in `tests/fixtures/legacy_maze.py`, and golden traces and replays pass
unchanged. Not done (not needed to meet the targets): column index for collision and
observation scans, lazy scroll, a per-seed reset cache.

## Oracle

`OraclePolicy` (`oracle.py`) is a `PrivilegedPolicy`: it reads the walls and
`Simulation.getScrollOffset()`, builds the periodic cell graph, runs BFS to a target
column 12 cells ahead (kept until the player is within 6 of it, so equal-length
routes do not flip-flop) and turns each move into one of the five actions, centring
the 10 px player in the 22 px corridor first. The real collision rules block it; it
never enters a wall (tested).

```bash
poetry run python -m training.oracle_stats --seeds train --max-steps 3000 --max-pace 5 \
    [--export-demos demos.jsonl]   # JSONL of {seed, tick, obs, action}
poetry run python -m training.evaluate --policy oracle --seeds validation
```

### Ceilings (train seeds 0-99, `--max-steps 3000`, `frame_skip` 1)

| start_pace | Mean score | Mean steps | Death rate | Mean final pace | Tile cleared | Clear steps | vs shortest path |
|---|---|---|---|---|---|---|---|
| 0 | 1519 | 2998 | 1% | 1.0 | 100% | 1371 | 1.06 |
| 1 | 1286 | 2525 | 18% | 1.8 | 84% | 1350 | 1.06 |
| 2 | 1079 | 2125 | 33% | 2.7 | 70% | 1323 | 1.04 |
| 3 | 773 | 1481 | 56% | 3.5 | 48% | 1245 | 1.00 |
| 4 | 600 | 1093 | 69% | 4.3 | 34% | 1143 | 0.98 |
| 5 | 371 | 652 | 86% | 5.2 | 20% | 1097 | 0.98 |

Findings:

- The oracle clears the first tile on **all 100 train seeds at pace 0**, in about 1.06x
  the shortest-path length. The one pace-0 death happens after the first pace
  increase (tick 1800).
- No start pace is sustainable for the whole horizon: from pace 1 the world scrolls
  faster than a perfect player can afford to backtrack out of dead ends (pace `p`
  scrolls `p` px every 10 ticks, the player moves 1 px per tick), so even perfect
  information dies. This is a property of the game, not a bug in the oracle. The
  "highest sustainable pace" is therefore pace 0 for about 1,800 ticks.
- Consequence for training: `start_pace` 0 with `max_steps` 3000 is the regime where
  the oracle is a meaningful ceiling. The plan's example `start_pace` of 4 would only
  ask the agent to survive a regime where the oracle itself dies 69% of the time.

## Environment

`MazeEnv(seeds, start_pace, max_steps, frame_skip, reward_config)` (Gymnasium):
`Discrete(5)` actions, `Box(0, 1, (89,), float32)` observations, reward and
termination from the shared `RewardFunction` / `Simulation`. `reset(seed=...)` seeds
Gymnasium's RNG, which draws the maze seed from `seeds` (one seed = fixed-seed
experiment); `options={"maze_seed": s}` forces one. `info` has `score`, `ticks`,
`pace`, `maze_seed`. It passes `check_env`, is deterministic for given seeds and
actions, and the reward sum equals the harness `total_return`.

## Training

```bash
poetry install --with ml-env,ml-train
poetry run python -m training.train --config training/configs/smoke.json        # ~10 s
poetry run python -m training.train --config training/configs/v1_fixed_seed0.json
poetry run python -m training.train --resume training/runs/v1_fixed_seed0       # continue
poetry run python -m training.expert --run training/runs/v1_fixed_seed0 --model best
```

A config (`training/configs/*.json`, committed; unknown keys are rejected; the
`test` seeds are refused) pins everything. A run writes `training/runs/<name>/`
(gitignored): `config.json` (copy), `metadata.json` (git commit and dirty flag,
library versions, Python, platform, seed), `metrics.csv`, `checkpoints/`, `best.zip`
(best mean return on `select_on`), `final.zip`, `eval.json`, and `replays/best` and
`replays/final` (best and worst episode per eval set, watch them with
`python -m training.overlay --replay <file>`). An existing run directory is never
overwritten: change a config by making a new one.

Every `eval_every_timesteps` the shared `evaluate()` runs through `ModelPolicy` with
deterministic actions on the training seed(s) and, separately, the `validation`
seeds. Seeds for numpy, torch, SB3 and the environments derive from `seed`; torch
uses one thread. Known nondeterminism: subprocess scheduling when `n_envs > 1`, and
`--resume` does not restore RNG state. An optional `curriculum` list of
`{"timesteps": N, "start_pace": p}` stages recreates the training environments.

## Expert ladder

`expert.ladder(model, heuristic, oracle)` (reports must share `start_pace`,
`max_steps`, `frame_skip` and seeds) returns the highest rung reached:

| Rung | Criterion |
|---|---|
| 1 | Mean return above the heuristic |
| 2 | Mean score >= `r2` x oracle mean score |
| 3 | Mean score >= `r3` x oracle mean score |
| 4 | Mean final pace >= `pace` with death rate <= `max_death_rate` |

Thresholds are `EXPERT_LADDER` in `config.py`: `r2` 0.5, `r3` 0.9, `pace` 1,
`max_death_rate` 0.1, set from the ceilings above. `python -m training.expert` re-runs
random, right, heuristic, oracle and the model on every eval set of the run and
writes `ladder.json`. Training-seed results are labelled as possibly memorised.

## First experiment (`v1_fixed_seed0`)

Settings: `start_pace` 0, `max_steps` 3000, `frame_skip` 4, seed 0 only for training,
reward `step_penalty` 0.001 / `death_penalty` 100 (`configs/rewards/v1.json`), PPO
with 8 subprocess environments, 2M timesteps. **Not yet run**: it is a long run, so
launch it yourself:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  poetry run python -m training.train --config training/configs/v1_fixed_seed0.json \
  2>&1 | tee training/runs/v1_fixed_seed0.log
```

Baselines at the same settings (`--max-steps 3000 --start-pace 0 --frame-skip 4`,
reward v1):

| Seeds | Policy | Mean return | Median | Mean score | Mean steps | Death rate |
|---|---|---|---|---|---|---|
| train (0-99) | random | -99.7 | -101.0 | 1.4 | 207 | 100% |
| train | right | -37.6 | -47.3 | 58.7 | 2374 | 94% |
| train | heuristic | 182.0 | 162.0 | 189.0 | 2987 | 4% |
| train | oracle | 1471.1 | 1511.0 | 1475.1 | 2998 | 1% |
| validation (1000-1049) | random | -96.4 | -101.0 | 4.8 | 391 | 100% |
| validation | right | -36.3 | -58.2 | 58.1 | 2362 | 92% |
| validation | heuristic | 197.0 | 184.0 | 200.0 | 3000 | 0% |
| validation | oracle | 1472.6 | 1543.0 | 1477.6 | 2967 | 2% |

Expectation: the fixed-seed agent should clear rung 1 quickly. If it stalls in the
first dead end, plain reward is not enough and oracle-based shaping (plan section 9)
is the next step. Held-out (validation) performance is expected to lag because the
observation has no memory; that is the next experiment, not a pipeline failure. Look
at the best and worst replays in the overlay before trusting any number.
