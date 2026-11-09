# Plan 4 — v2 Training: Multi-Seed and Oracle-Shaped Reward

Status: implemented and run (v2a, v2b finished at rung 0); results in training/README.md
Scope: fix the two causes of the `v1_fixed_seed0` stall (one training maze; a reward that pays for entering dead ends) and measure each fix separately. Everything is under `training/`; no change to `infinite_maze/`.
Prerequisites: [Plan 3](./03-ml-readiness-and-first-training-run.md) implemented (it is) and `v1_fixed_seed0` finished (it is).
Owner: `ml-dev`. Long runs are launched by the user.

## 1. What v1 showed

Run: `training/runs/v1_fixed_seed0`, 2M timesteps, PPO, seed 0 only, `max_steps` 3000, `frame_skip` 4, reward `v1`. Measured with `python -m training.expert --run ... --model final`.

| Seed set | Heuristic (return / score) | Model | Oracle | Ladder |
|---|---|---|---|---|
| train (seed 0) | 118 / 121 | 162 / 165 | 1145 / 1148 | rung 1, 14% of oracle |
| validation (50 seeds) | 197 / 200 | 97 / 134, 34% deaths | 1473 / 1478 | rung 0, 9% of oracle |

Findings:

1. **Pipeline works.** Training return went from -101 to +162 with no deaths, and every artifact (metrics, checkpoints, replays, metadata, ladder) was produced.
2. **It memorised a stall.** Seed 0 return reached 162 at 401k steps and never improved. The agent sits in a dead end (score 165) while the oracle scores 1148 on the same maze. Rung 1 on train is therefore weak evidence: the heuristic is stuck in the same dead end one cell shallower.
3. **Held-out is below the heuristic.** The model beats the heuristic on 7 of 50 validation seeds, and 33 of 50 reach score >= 100 (heuristic: 50 of 50). Validation was still improving at the end (return -13 at 100k steps, about 97-100 at 2M; death rate 84% to 34%).
4. **`best.zip` is meaningless.** `select_on` was `train`, a single seed, so `best` is the first model to hit 162.0 and later ties never replaced it. It scores 1.2 on validation against 97 for `final.zip`.
5. **Why the reward misleads** ([reward.py](../training/reward.py)): progress is `+1` per pixel moved right, so entering the dead end at spawn pays about `+22` per cell and the correct route (5 cells down first, no x progress) pays `0`. Backing out of the dead end costs `-22` per cell. The task's true optimum is still the oracle path (the base reward is a net x displacement), but credit assignment and a 4.5-cell window cannot find it.

## 2. Decisions adopted

| # | Decision | Choice | Reason |
|---|---|---|---|
| D5 | What changes in v2 | Two changes, run as **two configs** so each is measurable: **v2a** multi-seed, no shaping (control); **v2b** multi-seed plus oracle shaping | One change at a time is not possible (v1 already showed single-seed is not the answer), but a control isolates the shaping effect |
| D6 | Shaping form | **Static potential** `Φ(s) = -weight * d(cell)`, `d` = BFS cell distance to a far column, computed once per episode | See section 3: the maze is periodic and fixed in world coordinates, so one field is valid for the whole episode and `Φ` never jumps. The oracle's `_distance_field` retargets every few cells, which would break the potential |
| D7 | Terminal handling | **No shaping term on the terminating transition**; `death_penalty` alone punishes death | The textbook `Φ(terminal)=0` convention with `Φ <= 0` pays `+weight*d` for dying (a reward hack). With `Φ >= 0` it makes late deaths cost thousands. Skipping the term mirrors how the base progress reward already behaves. A test must prove dying is never rewarded |
| D8 | Model selection and reporting | `select_on: "validation"`; report **oracle fraction and mean score** on validation as the headline; `test` seeds are used once, at the very end of the project | Fixes finding 4. Return is reward-config dependent, score is not |
| D9 | Everything else | Keep v1's PPO hyper-parameters, observation, `frame_skip` 4, `max_steps` 3000, `start_pace` 0 | Do not tune several things at once |
| D10 | Evaluation reward | Evaluation and baselines keep the **unshaped** `reward` config, so tables stay comparable with v1 and the README | Shaping is a training aid only |

Out of scope (v3 candidates, section 8): memory or frame stacking, larger observation window, behaviour cloning, pace curriculum, more algorithms.

## 3. Design

### 3.1 Why a static field is valid

Walls are fixed in world coordinates: scrolling moves them left and recycling moves a passed wall one tile period (`2 * MAZE_ROWS` columns = 30 columns) to the right, so the maze is periodic and unchanging ([simulation.py](../infinite_maze/core/simulation.py) `getScrollOffset` docstring; [oracle.py](../training/oracle.py) `MazeGeometry`). A BFS over the unrolled columns from spawn to a far column therefore stays correct for the whole episode.

Prototype checked while writing this plan (throwaway script, not committed), seeds 0, 1, 2, `MazeGeometry` with 30 columns x 19 rows, BFS to column 150:

| Measurement | Result |
|---|---|
| BFS time | 2-3 ms per episode (reset is about 6 ms already) |
| Cells covered, none unreachable | 2,983 (19 rows x 157 columns) |
| Distance from spawn cell `(-1, 8)` to column 150 | 276 / 252 / 321 cells |
| Furthest column the oracle reached in 3000 ticks | 52 / 81 / 61 |
| Seed 0 distances: spawn, then right along row 8 | 276, 277, 278, 281 (rising: dead end) |
| Seed 0 distances: spawn, then down the spawn column | 275, ..., 271, then 270 at `(0, 13)` (falling: correct route) |

So `Φ` gives the right sign for exactly the decision v1 got wrong.

### 3.2 New module `training/shaping.py` (stdlib only)

```python
@dataclass(frozen=True)
class ShapingConfig:
    weight: float = 0.0   # reward per cell of path distance; 0 disables shaping
    gamma: float = 0.995  # must equal ppo.gamma (validated in ExperimentConfig)

class PotentialField:
    """Static cell-distance field for one episode, read from a fresh Simulation."""
    def __init__(self, sim: Simulation, max_steps: int) -> None: ...
    def potential(self, sim: Simulation) -> float: ...   # -d(cell); 0.0 never silently
```

- Reuse `MazeGeometry` and the BFS neighbour rules from [oracle.py](../training/oracle.py). Promote `OraclePolicy.current_cell` to a `MazeGeometry` method (or a module function) so shaping does not instantiate an `OraclePolicy`. Keep oracle behaviour and its tests unchanged.
- Target column `T = start_column + ceil(max_steps * player_speed / MAZE_CELL_SIZE) + 4`. The player moves at most 1 px per tick in world x, so `T` is unreachable within `max_steps` and the player can never leave the field. Lower bound of the window: `start_column - 6`.
- If `potential()` is asked about a cell outside the field it must raise (fail loud). Do not return 0.

### 3.3 Where shaping is applied

In `MazeEnv.step` ([env.py](../training/env.py)), once per **macro step** (after `repeat_action`), not per tick, so the discount `gamma` matches PPO's per-decision `gamma`:

```text
F = gamma * Φ(s_after) - Φ(s_before)        # not terminated
F = 0                                        # terminated (D7)
reward = (base_reward + F) * train_reward_scale
info["base_reward"], info["shaping_reward"]   # unscaled, for logging and tests
```

- `MazeEnv(..., shaping: Optional[ShapingConfig] = None, reward_scale: float = 1.0)`. With the defaults, behaviour is byte-identical to today (the existing "reward sum equals harness `total_return`" test must still pass).
- Build the `PotentialField` in `reset()` after the simulation is reset.
- `evaluate()`, `ModelPolicy`, `evaluate_sets` in [train.py](../training/train.py) and the expert ladder do **not** receive shaping. Only `make_training_env` does.
- `Φ` is never part of the observation. It is privileged information used only in the training reward, so the policy cannot read it at test time.

### 3.4 Weight and scale

- **`weight = 44` (2 x cell size).** Per cell, with the base progress of 22 px per cell: entering a dead end nets `+22 - 44 = -22`; backing out nets `-22 + 44 = +22`; the correct detour with no x progress nets `+44`. With `weight = 22` backing out would net 0 and give no incentive to leave a dead end. This is the number to defend in the smoke run, not a tuned value.
- **`train_reward_scale = 0.05` (new top-level field in `ExperimentConfig`, default 1.0).** Shaping adds thousands of return over an episode (oracle path is on the order of 100 cells, so on the order of `44 * 100`). Scaling keeps returns near v1's magnitude so the value function can fit. It does not change the optimal policy, and PPO's advantage normalisation keeps `ent_coef` meaningful. Apply it to **both** v2 configs so the control differs only by shaping.
- If the smoke run shows `explained_variance` stuck near 0 or exploding `value_loss`, lower the scale before changing anything else.

### 3.5 Config changes ([experiment.py](../training/experiment.py))

- Add `shaping: ShapingConfig` and `train_reward_scale: float = 1.0` to `ExperimentConfig`.
- `from_dict` rejects unknown keys in `shaping`, like `reward` and `ppo`.
- Validate: `shaping.gamma == ppo.gamma`; `shaping.weight >= 0`; `train_reward_scale > 0`.
- Old `config.json` files (including `v1_fixed_seed0`) must still load and give identical behaviour.
- `make_training_env` in [train.py](../training/train.py) passes `shaping` and `train_reward_scale` to `MazeEnv`; `collect_metadata` records them.

## 4. Work packages

```text
WP1 shaping module ──► WP2 env + config wiring ──► WP3 reward-ordering gate ──► WP4 smoke ──► WP5 v2a/v2b runs ──► WP6 report
```

### WP1 — `PotentialField` and `MazeGeometry` refactor

Section 3.2 only. Acceptance:

- Unit tests on hand-built or seeded mazes: `d` decreases by exactly 1 per cell along the oracle's path from spawn; on seed 0 `d(0,8) > d(-1,8)` and `d(-1,9) < d(-1,8)`.
- **Invariance test:** the field built at `reset()` equals a freshly built BFS after N ticks of scrolling and recycling, for the same world cells.
- Window guard: a cell outside the field raises.
- Existing oracle tests pass unchanged.

### WP2 — Env and config wiring

Sections 3.3 and 3.5. Acceptance:

- Shaping off, scale 1.0: identical rewards to today (existing env tests unchanged).
- `info["base_reward"] + info["shaping_reward"]` times scale equals the returned reward.
- **No reward for dying:** on a terminated step the shaping term is exactly 0 and the total is below that of a non-terminated step from the same state.
- Determinism: same seeds and actions give identical shaped rewards across two runs.
- `check_env` still passes. Config tests: `v1_fixed_seed0.json` loads unchanged, `gamma` mismatch and unknown keys are rejected, `test` seeds still refused.

### WP3 — Reward-ordering gate (before any training)

A cheap functional test or script that evaluates random, right, heuristic and oracle on at least 20 `train` seeds with shaping on and reports the **training** return. Required ordering: oracle > heuristic > right > random. If the oracle does not clearly lead, shaping is wrong; stop and fix before spending compute. Record the numbers in `training/README.md`.

### WP4 — Smoke run

New `training/configs/smoke_v2.json`: shaping on, 100 train seeds, a few thousand timesteps, about 10 s like `smoke.json`. The agent may run this. Check that it completes, that `metrics.csv` has sane `value_loss` and `explained_variance`, and that evaluation still reports unshaped returns.

### WP5 — The two runs (user launches)

Both configs: `train_seeds: "train"` (0-99), `eval_seeds: {"train": <first 20 seeds as an explicit list>, "validation": "validation"}`, `select_on: "validation"`, `start_pace` 0, `max_steps` 3000, `frame_skip` 4, `n_envs` 8, `total_timesteps` 5,000,000, `eval_every_timesteps` 250,000, reward = v1 values, PPO = v1 values, `train_reward_scale` 0.05, `seed` 0.

| Config | `shaping.weight` |
|---|---|
| `training/configs/v2a_multiseed_control.json` | 0 |
| `training/configs/v2b_multiseed_shaped.json` | 44, `gamma` 0.995 |

Cost: v1 took about 21 minutes for 2M timesteps with 8 environments, so each run is roughly an hour including evaluation. Run them one after the other so the box is not oversubscribed. Evaluation seeds are fixed; do not change them between runs.

### WP6 — Report

Run `python -m training.expert --run training/runs/<name> --model best` for both (`best` is now chosen on validation). Add a results section to [training/README.md](../training/README.md) in the style of the v1 tables (heuristic, oracle, v2a, v2b on validation, plus train-set 20), a section describing shaping, and the replays reviewed in the overlay. State what stalled and where.

## 5. Success criteria (stated so the outcome can be judged)

Measured on the 50 validation seeds, `final` or `best`:

| Outcome | Criterion | Meaning |
|---|---|---|
| Minimum | v2b mean return > heuristic (197) and death rate <= 10% | Ladder rung 1 on held-out seeds |
| Target | mean score >= 0.5 x oracle (about 739) | Ladder rung 2 |
| Shaping works | v2b beats v2a by a clear margin on validation mean score | Shaping, not extra seeds, was the missing piece |
| Shaping not needed | v2a is within noise of v2b | Drop shaping (less machinery) and look at observation or memory |
| Reward bug | v2b train-20 score < about 400 | Do not trust the run; check WP1-WP3 before anything else |
| Partial observability | v2b train-20 score >= 0.5 x oracle but validation score < 0.2 x oracle | The policy can solve mazes it has seen but cannot see dead ends in a 4.5-cell window: go to section 8 |

One run per config is a single sample. If v2a and v2b differ by less than the spread visible in `eval.json` between consecutive evaluations, repeat both with `seed` 1 and 2 before concluding anything.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Shaping rewards dying or oscillating | D7 plus the WP2 test; the WP3 ordering gate; review best and worst replays in the overlay |
| Large state-dependent value offset (`-Φ(s)` is not observable from the local window) | Reward scale; watch `explained_variance`; a constant per-episode shift of `Φ` is allowed if needed (it does not change the optimum) |
| `gamma` mismatch between shaping and PPO | Config validation |
| Field window too short for `max_steps` | `T` derived from `max_steps`; out-of-field cell raises |
| Truncation at `max_steps` mistaken for termination | Truncation keeps the normal `F`; SB3's vec envs pass `TimeLimit.truncated` so PPO bootstraps. Confirm in WP2 by reading the SB3 version in use |
| Selecting on validation slightly overfits to it | `test` seeds untouched until the final report; fixed validation set |
| Single-run noise | Section 5 repeat rule |
| Return comparisons across configs | Headline metrics are score and oracle fraction; evaluation reward stays unshaped (D10) |

## 7. Definition of done

- WP1-WP3 tests added and passing, and CI parity commands pass: `black`, `flake8`, `mypy` on `infinite_maze/ training/`, and `pytest -m "not slow and not performance"`.
- Core install and CI without the ML groups unaffected.
- `v2a` and `v2b` completed from committed configs with replays, metrics and ladder reports.
- `training/README.md` updated with shaping, the ordering-gate numbers and the v2 results, including a plain statement of which success row in section 5 was hit.
- `infinite_maze/` unchanged.

## 8. Next candidates (only after v2 results)

Triggered by section 5's "partial observability" row, or if both runs plateau near the heuristic:

1. Memory: frame stacking or recurrent PPO (`sb3-contrib`); needs a new optional dependency and a Python-version check.
2. Observation: a wider raster, or an explicit "recently visited cell" channel.
3. Behaviour cloning from `python -m training.oracle_stats --export-demos`, then PPO fine-tuning.
4. Pace curriculum, once the agent reaches the first pace increase reliably (the oracle itself dies from pace 1, see the README ceilings).
