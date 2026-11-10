# Plan 4 — v2 Training: Multi-Seed and Oracle-Shaped Reward

Status: v2 and v3 run (all rung 0); learning, not visibility, is the limit; next steps in sections 13-14
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

## 8. Next candidates (superseded by sections 9 and 10)

Original list, written before the v2 results. Section 10 orders and tests these.

1. Memory: frame stacking or recurrent PPO (`sb3-contrib`); needs a new optional dependency and a Python-version check.
2. Observation: a wider raster, or an explicit "recently visited cell" channel.
3. Behaviour cloning from `python -m training.oracle_stats --export-demos`, then PPO fine-tuning.
4. Pace curriculum, once the agent reaches the first pace increase reliably (the oracle itself dies from pace 1, see the README ceilings).

## 9. Outcome of v2 (measured; details in [training/README.md](../training/README.md))

Validation, 50 seeds, `best` (selected on validation); heuristic 200 score, oracle 1478.

| Run | Score | Death rate | Oracle fraction | Ladder |
|---|---|---|---|---|
| v2a multi-seed, no shaping | 190 | 6% | 12.9% | rung 0 |
| v2b multi-seed + shaping | 144 | 10% | 9.8% | rung 0 |

Section 5 rows hit: no minimum or target; v2b train-20 score 132 is below the 400 "reward bug" threshold; v2a is not worse than v2b, so **shaping did not help**. One run per config at `seed` 0.

What the diagnostics showed:

1. All models, and the heuristic, stop after about 8 path cells (first dead end, score about 200); the oracle gains 123. Stochastic and deterministic evaluation score the same, so it is not an argmax artefact.
2. v2a plateaued at the heuristic from about 600k steps with healthy value metrics (`value_loss` about 0.1, `explained_variance` 0.9). More steps and more seeds did not help, so **the limit is information, not data or compute**. This is the best-supported finding, but it is still a hypothesis: no experiment has yet changed the observation.
3. v2b's `value_loss` is 40-1000 and its return is dominated by the discount-leak term `weight * (1 - gamma) * d` of the potential, which the observation cannot see. Plausible cause of its being worse than the control; not isolated by an experiment.
4. The WP3 ordering gate passed but did not predict learnability. It ranks whole-episode discounted returns only.

Not yet done: the best and worst replays were not watched in the overlay.

## 10. Remediation: test the information hypothesis cheaply, then fix it

Principle: stop spending hour-long PPO runs until a minutes-long supervised probe says which observation can solve the mazes. The oracle can label any state (`OraclePolicy.act_privileged(sim)` needs only a `Simulation`), so supervised learning from oracle actions is a fast, deterministic proxy for "is the answer in this observation?".

### 10.1 Decisions

| # | Decision | Choice | Reason |
|---|---|---|---|
| D11 | Shaping | Shelve. Keep the code and the committed v2b config; do not run it again | It trained worse than its control and added a large unobservable return term |
| D12 | First experiment | Supervised probe (behaviour cloning plus DAgger) on oracle labels, comparing observation sizes, before any new PPO run | Minutes per experiment versus about an hour; separates "cannot see" from "cannot learn" |
| D13 | Observation change | Make the observation configurable per run (`ExperimentConfig.observation`), default equal to today | Old runs, models and the ladder must keep working; a model is only valid with the observation it was trained on |
| D14 | Memory | Fallback only, after the probe says a wider window is not enough | Needs more machinery (visited channel, or `sb3-contrib` recurrent PPO and a Python-version check) |
| D15 | Reward and PPO | Keep v1 reward, `frame_skip` 4, `max_steps` 3000, `start_pace` 0, v1 PPO values | One change at a time |
| D16 | Observation stays non-privileged | The observation may only contain what a human sees on screen: the wall layout in a window around the player, rays and scalars. No path distances, no goal direction, no oracle output, no potential field | Guards against the agent becoming a thin wrapper around the oracle (owner concern). A wider window is not leakage: the screen is 640 x 480 px and shows far more than 4.5 cells, so the window is capped at what is rendered |
| D17 | Role of the oracle | Teacher for the probe (R1-R2) and optional initialisation (R4); the final agent is judged on a reward-trained PPO run whose observation obeys D16. The oracle is never queried at evaluation or during the final training reward | Imitation alone caps the agent at the oracle and copies its blind spots; reward training can exceed it (the oracle itself dies from pace 1) |
| D18 | Training seeds | If G1 shows overfitting, add a fixed `train_extra` seed set (3000-3999, training only) to `config.SEED_SETS`; `train` 0-99, `validation` and `test` do not change | Owner agreed; keeps every existing table comparable |

### 10.2 Work packages

```text
R1 probe tooling ──► R2 window sweep ──► gate G1 ──► R3 configurable observation ──► R4 PPO from BC init ──► R5 report
                                              └─ fail ──► R6 memory / hint spike
```

**R1 - Probe tooling** (new `training/probe.py`, needs `ml-train`; no new dependency, torch is already there).

- Demo export must match training: `python -m training.oracle_stats --export-demos ... --frame-skip 4` (the default is 1, which would label states the agent never sees), `--max-pace 0`, train seeds only. Never export from `validation` or `test` seeds.
- Behaviour cloning: fit a small MLP (same `[64, 64]` shape as PPO) with cross-entropy on `(obs, action)`.
- DAgger rounds (2-3): run the current learner, label every state it visits with `act_privileged`, add to the dataset, refit. Plain cloning only sees oracle-visited states, so it fails off-distribution regardless of observability; the probe must not be blamed on the observation for that.
- Report per run: action accuracy overall and at **decision states** (states where the oracle's action is not "continue the current direction"), plus `evaluate()` on validation with the shared harness (`frame_skip` 4).
- Save as an SB3-compatible `ModelPolicy` or a small `Policy` so `training.expert` can rank it unchanged.
- Acceptance: a test that the probe's policy plays through `evaluate()`, deterministic for a fixed seed; the existing suite and CI parity commands stay green.

**R2 - Window sweep** (an experiment, not code beyond R1).

Observation raster sizes 9 (today), 15, 21 and 31 cells of 11 px (31 x 11 = 341 px, inside the 640 x 480 screen, so D16 holds; do not go beyond the rendered area), with and without the ray and scalar features, same demo seeds, same DAgger rounds. `OBS_GRID_SIZE` and `OBS_GRID_CELL_PX` are module constants today; for the sweep pass them through `build_observation(sim, grid_size=...)` (parameters already exist) and an observation size argument, without touching defaults.

Gate **G1** on validation score of the DAgger-trained probe (heuristic 200, oracle 1478):

| Result | Reading | Next |
|---|---|---|
| Some window >= 600 (about 40% of oracle) | The answer is in a wider window | R3 with the smallest window that reaches it |
| All windows <= 250, accuracy at decision states near chance | Memoryless local views cannot disambiguate | R6 |
| Large windows fit train seeds but validation stays < 250 | Overfitting to 100 mazes, not observability | Add `train_extra` (D18) and re-probe |

**R3 - Configurable observation** (only after G1 passes).

- `ExperimentConfig.observation: ObservationConfig(grid_size, cell_px)`, unknown keys rejected, default 9 and 11 so every existing config and `config.json` loads unchanged (same test pattern as the shaping fields).
- Thread it through `MazeEnv`, `ModelPolicy`, `evaluate_sets`, `training.expert`, `oracle_stats --export-demos` and the overlay (which reads `config.OBS_GRID_SIZE` directly, `overlay.py` lines near 98). `observation_size()` must take the config; a model must refuse to load into a mismatched observation size with a clear error.
- Record it in `metadata.json`. Acceptance: shaping-off, default-observation rewards and observations byte-identical to today; `check_env` passes; golden and oracle tests unchanged.

**R4 - PPO from a BC start** (long run, user launches; one config, `v3a_bc_init`).

Initialise PPO's policy from the DAgger network (matching architecture), unshaped v1 reward, multi-seed `train`, `select_on: "validation"`. Add a 0.5-1M step control without BC init only if the budget allows, so the BC effect is measurable. Risk to watch: PPO can unlearn the BC policy early (`ent_coef` and learning-rate warmup are the levers; do not tune beyond one adjustment). Success criteria are those of section 5, unchanged.

**R5 - Report.** README results in the v1/v2 table style: probe table (window x score x decision-state accuracy), the G1 outcome, and the replays actually watched in the overlay, including the best and worst of each run.

**R6 - Fallback if G1 fails: memory or a hint spike.**

- First, a one-hour diagnostic that adds the oracle's next-move direction as an observation feature. This is an upper-bound experiment only: it lives in the probe CLI behind an explicit flag, is absent from `ExperimentConfig`, and no committed config, run or reported result may use it (D16). If the probe and PPO both solve the mazes with it, the information is the only gap and a "recently visited cells" channel (egocentric raster of visited maze cells, tracked in a shared wrapper so env, `ModelPolicy` and `evaluate` agree) is the direct fix. If they still fail, the problem is learning or reward, not observation.
- Recurrent PPO (`sb3-contrib`) only after that; it needs a new optional Poetry group and a Python-version check (declared `^3.8.1`, ML groups already `>=3.10`).

### 10.3 Things that must not happen

- Any privileged input in a training or evaluation observation (D16), or oracle queries outside the probe and the BC initialisation (D17). Add a test that `build_observation` output depends only on walls and the player on screen (same observation for two simulations whose walls differ only outside the window), and that `ExperimentConfig` has no switch for privileged features.
- Validation or `test` seeds in demo export, probe training or DAgger labelling. `test` stays untouched until the final report.
- A change to evaluation reward, `frame_skip`, `max_steps` or the baselines between experiments; tables must stay comparable with v1 and v2.
- Declaring the information hypothesis proven from one probe: confirm with the R4 PPO result and one repeat at another `seed`.

### 10.4 Definition of done for this remediation

- R1-R2 delivered with tests, CI parity commands green, core install unaffected, `infinite_maze/` unchanged.
- G1 outcome recorded in `training/README.md` with the numbers, even when negative.
- If G1 passes: R3 and R4 complete, success row from section 5 stated plainly, together with the **independence check**: the PPO model's score versus the BC-only probe and versus the oracle, and its behaviour on a regime the oracle cannot handle (`start_pace` 1 for a short horizon). A model that only ever matches the probe is a distilled oracle, not a learner; say so if that is what the numbers show.
- If G1 fails: R6 diagnostic complete and the next plan written from its result.

### 10.5 Owner decisions (resolved)

1. Probe-first order (about a day of work) before any wider-observation PPO run: accepted.
2. A larger observation vector (up to a 31 x 31 raster, about 960 floats) is accepted. The owner's concern is that the agent becomes the oracle in disguise; D16 and D17 and the independence check in 10.4 answer it: wider means seeing more of the screen, not receiving the solution.
3. Extending training seeds if the sweep shows overfitting: accepted (D18).

## 11. Progress log: remediation R1 done, R2 in progress (stopped for the day)

Delivered (CI parity green: black, flake8, mypy, `pytest -m "not slow and not performance"` 1220 passed; `infinite_maze/` unchanged):

- `training/probe.py`: BC + DAgger probe (`python -m training.probe --grid-size N [--no-extras] [--arch mlp|cnn] [--sweep]`), `ProbePolicy` through the shared `evaluate()` (new `Policy.grid_size`/`cell_px` attributes, defaults unchanged), parallel rollouts/evaluation, `--horizon` ceiling mode, D16/D17 safeguards (seed checks, tests that observations ignore walls outside the window). Tests: `tests/unit/test_training_probe.py`.
- D18 applied early: `train_extra` seeds 3000-3999 in `training/config.py` (`SEED_SETS`); other sets unchanged. Probe fits on `train_extra` 3000-3399, holds out train 0-19 for accuracy, plays validation only.
- Probe constants live in `training/config.py` (`PROBE_*`). `training.probe --sweep` runs the full R2 grid but has **not been run**: the pilot results below show it would not be conclusive as designed.

Pilot findings (single runs, not a sweep; `training/runs/probe/` holds nothing yet):

| Experiment | Result |
|---|---|
| MLP 64x64, g15, fit on 80 mazes | Fit-seed score 46, validation 19: memorises mazes (so D18 was triggered) |
| MLP, g15 / g31, fit on 400 `train_extra` mazes, 4 DAgger rounds | Validation score 11 / 19, deaths 98-100%. Learner policies do not even reach the heuristic |
| Supervised accuracy on **oracle-visited** states of held-out mazes, decision states (majority class 36%) | MLP g15 71%, g31 62-67%; CNN g15 72%, g31 68%. 400 to 1000 mazes moves it only 71 to 73% (g15). A wider window and a CNN did not help; train accuracy reaches 96% (overfit) |
| 22% of oracle labels are pixel-level lane keeping, 78% cell moves | Not the dominant noise source |
| `--horizon H`: perfect planner that knows all walls but plans H columns ahead, validation score | H=12: 1478 (oracle), 8: 1299, 4: 1091, 2: 938 |

Reading, with caveats:

1. The horizon result says a perfect planner needs very little look-ahead (63% of the oracle at 2 columns), which argues **against** "the window is too small" as the main cause. It is not a clean window test: the planner's BFS still sees a full maze tile behind it and keeps a committed target (memory). A planner restricted to the window's columns and rows is the missing control.
2. Learned probes fail to generalise across mazes even when the window is large, so the bottleneck is more likely sample efficiency / inductive bias (MLP and CNN on a flattened raster, 400-1000 mazes) or label inconsistency than raw information. Not proven.
3. G1 as written (validation score of the DAgger probe) is not informative at this quality level: every window scores below 50. Do not read the sweep as "memoryless views are insufficient" yet.

Next steps (tomorrow):

1. Add a window-restricted planner (BFS only over cells inside the g x g window, no memory of the committed target) to `--horizon` and sweep g in 9, 15, 21, 31, 45, 55 (cap 55: 605 px stays on screen). This separates information from learning cleanly and decides G1.
2. If the window-restricted planner scores well at small g: the problem is learning. Candidates: far more mazes (up to all 1000 `train_extra`), data augmentation (vertical flip with up/down swap is not valid for the maze; check before using), a better probe target (cell-level direction), or reward-trained PPO with BC initialisation anyway.
3. If it scores poorly: the problem is information, so go to R6 (visited-cells channel / memory).
4. Only then rewrite G1 with the right measure and run `--sweep`.

## 12. Progress log: G1 decided, R3 done, v3 runs ready to launch

CI parity green (black, flake8, mypy; `pytest -m "not slow and not performance"` 1235 passed; slow training tests 6 passed). `infinite_maze/` unchanged.

**G1 (information ceiling).** New `training/windowed.py`: a memoryless planner that knows the walls exactly but may only use cells whose centres lie inside the g x g raster window and keeps no state between decisions (`python -m training.probe --window 9 15 21 31 45 55`, validation seeds, `frame_skip` 4, `max_steps` 3000, `start_pace` 0):

| Window (cells of 11 px) | Window (px) | Validation score | Death rate | Oracle fraction |
|---|---|---|---|---|
| 9 (today) | 99 | 174 | 36% | 12% |
| 15 | 165 | 218 | 24% | 15% |
| 21 | 231 | 504 | 18% | 34% |
| 31 | 341 | 839 | 18% | 57% |
| 45 | 495 | 1327 | 2% | 90% |
| 55 | 605 | 1268 | 4% | 86% |

Reading: **G1 passes on information.** With the current 9-cell window even a perfect memoryless planner scores about the heuristic (174 vs 200) - the same stall v1 and v2 showed - so the stall is mostly a visibility limit, not a data, reward or compute limit. Information grows steadily with the window and is close to the oracle at 45 cells. This is a ceiling for a perfect reader of the window, not a promise that PPO learns it: the earlier DAgger probe (section 11) generalised poorly at 15 and 31 cells, so learnability at large windows is still open. Caveats: cells are judged by their centre being in the window, so a wall edge half outside is treated as known; 45 cells is 495 px tall, 15 px more than the 480 px screen, though the maze itself is only 418 px tall (the extra rows are empty raster).

**R3 done (configurable observation).** `ExperimentConfig.observation: ObservationConfig(grid_size, cell_px)`, default 9 and 11 (every old config and `config.json` loads unchanged; windows wider than the 640 px screen are rejected). Threaded through `MazeEnv`, `ModelPolicy` (raises on an observation-size mismatch), `build_policy`, `evaluate` CLI, `training.expert`, `train.py` (env, evaluation, `metadata.json`), `oracle_stats --export-demos` and the overlay (`--grid-size`, `--cell-px`). Default-observation behaviour is unchanged (existing env, golden and oracle tests pass). Cost: env throughput about 2200 / 1870 / 1690 decisions/s at 9 / 31 / 45 cells.

**R4 prepared, not run (user launches).** Two configs differing only in window: `v3a_window31.json` and `v3b_window45.json`. Both: PPO from scratch with v1 reward and PPO values (D15), no shaping, `train_reward_scale` 0.05 as in v2, `train_seeds: "train_extra"` (1000 mazes, D18), `eval_seeds` = `train_extra` 3000-3019 (memorisable, labelled so) and `validation`, `select_on: "validation"`, 5M timesteps, `n_envs` 8. `smoke_v3.json` (about 10 s) passes. BC initialisation is **deferred**: the DAgger probe is too weak to be a useful start, and from-scratch PPO makes the window effect cleaner to read.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy poetry run python -m training.train \
  --config training/configs/v3a_window31.json 2>&1 | tee training/runs/v3a_window31.log
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy poetry run python -m training.train \
  --config training/configs/v3b_window45.json 2>&1 | tee training/runs/v3b_window45.log
poetry run python -m training.expert --run training/runs/v3a_window31 --model best
poetry run python -m training.expert --run training/runs/v3b_window45 --model best
```

Report back: the ladder tables, `eval.json` history, and `explained_variance` / `value_loss` from `metrics.csv`. Expected, as hypotheses to be tested: v3a above the 200 heuristic stall if PPO can read a 31-cell window (windowed ceiling 839); v3b higher still (ceiling 1327). If both stay near 200 the limit is learning, and the next steps are a convolutional feature extractor for the raster (SB3 custom extractor, still `ml-train` only) or more training mazes. Independence check (10.4): compare scores with the windowed ceiling and run the best model at `start_pace` 1 on a short horizon.

Remaining from section 10: R5 report after the runs, and the best/worst replays still need watching in the overlay (`python -m training.overlay --replay <file> --grid-size 31`).

## 13. Outcome of v3 (measured): wider windows did not help PPO; the limit is learning

Setup: `v3a_window31` / `v3b_window45`, PPO from scratch, `train_extra` (1000 mazes), v1 reward, no shaping, 5M timesteps, seed 0, one run each, `best` selected on validation. Ladder: **rung 0 on both sets for both runs.**

| Run | Validation score | Death rate | Oracle fraction | Train-extra-20 score | Windowed-planner ceiling |
|---|---|---|---|---|---|
| heuristic | 200 | 0% | 13.5% | 204 | |
| v2a (9 cells, 100 mazes) | 190 | 6% | 12.9% | 193 (train 0-19) | 174 |
| v3a (31 cells) | 172 | 16% | 11.6% | 153 | 839 |
| v3b (45 cells) | 141 | 32% | 9.5% | 131 | 1327 |
| oracle | 1478 | 2% | 100% | 1358 | |

Diagnostics on the 50 validation seeds, `best.zip`: path cells gained 7.1 (v3a) and 6.8 (v3b), against 7.9 for v2a and 123 for the oracle; episodes with no progress for more than 1000 ticks: 24/50 (v3a) and 12/50 (v3b), deaths 8 and 16 of 50. Training and validation scores are close, so this is not memorisation. Training metrics are healthy (`value_loss` 0.05-0.2, `explained_variance` 0.6-0.9), and v3a's training score was still creeping up (125 at 0.5M to 182 at 5M) with `eval.json` noisy between evaluations (v3b swings 43 to 141).

Reading (not tested beyond this):

1. The information hypothesis is half confirmed: visibility sets a ceiling (9 cells cannot beat the heuristic even for a perfect reader), but **removing that ceiling did not help PPO**. With 31 and 45 cells the ceilings are 839 and 1327 and the agents reach 172 and 141, so the gap is a learning or representation problem, not information.
2. The same conclusion came from the probe: a flat MLP on a 961 or 2033-float raster did not generalise across mazes in supervised learning either (decision-state accuracy 62-73%, no gain from window size or from a CNN at 400 mazes). Larger inputs made the MLP policy slower to learn, not better.
3. v3a is slightly better than v3b and both are worse than v2a: more input dimensions with the same network and budget is a cost, not a benefit.
4. Untested contributors: the policy network is a `[64, 64]` MLP on a flattened image (no translation structure), the reward still pays `+22` per cell into dead ends (the credit-assignment problem from section 1, now with a vision that could avoid it), 5M steps may be too few for a 1000-maze distribution, and a single seed.

## 14. Next steps (proposal; decide before running anything)

The cheap lever left that the data already points to is a better learner for the same observation, tested in the supervised probe first, since it takes minutes:

1. **Probe a convolutional policy on large windows with far more data** (all 1000 `train_extra` mazes, `--arch cnn`, window 31 and 45). The CNN probe at 400 mazes reached 68-72% on held-out decision states and 96% on training states: it overfits, so data and regularisation (dropout or weight decay, crops that keep the player at the centre but vary nothing else) are the experiments, not size. Decide with the **windowed planner as the target**: a probe that clones the windowed planner (labels from `WindowedPlanner`, not the full oracle) removes label noise caused by plans the window cannot see, which is plausibly why the oracle-label probe generalised badly. That is a small change in `training/probe.py` (a `teacher` option) and the first thing to try.
2. If a cloned policy reaches a validation score of 500 or more at 31 cells, use it to initialise PPO (R4 as originally written) and fine-tune with the v1 reward; compare with a no-init control for 0.5-1M steps.
3. If it does not, add the SB3 custom CNN feature extractor for PPO from scratch (`ml-train` only) and repeat v3a, still one change at a time. Reward shaping stays shelved (D11) until a learner that can read the window exists.
4. Whatever runs: repeat the best configuration at `seed` 1 and 2 before concluding, and still watch best and worst replays with `python -m training.overlay --replay <file> --grid-size 31`.

Not recommended now: more PPO steps on the same setup (v3a's slow upward trend is not a path to the 839 ceiling), or a 55-cell window (the ceiling is flat past 45).
