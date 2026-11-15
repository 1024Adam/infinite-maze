# Plan 5 — Cell-Level Learnability: Separate Control from Navigation

Status: approved, not started
Scope: find out, conclusively and cheaply, whether PPO can learn to navigate these mazes once pixel-level lane keeping is removed from the problem, and carry the result back to the real game. Everything is under `training/`; no change to `infinite_maze/`.
Prerequisites: [Plan 4](./04-v2-multi-seed-and-shaped-training.md) finished (v1-v5a all at rung 0; its section 18 lists the options this plan replaces).
Owner: `ml-dev` implements; the user launches runs longer than 30 minutes.

## 1. Why the previous attempts stalled

Seven training runs and two cloning probes all land at rung 0 (validation score 130-190; heuristic 200, oracle 1478). Each changed one of maze count, window, network or reward. **None changed the action interface**, and the evidence says that is the common cause.

| Finding | Evidence |
|---|---|
| Exploration at pixel level is close to impossible | 5 actions, 1 px per tick, `frame_skip` 4: about 750 decisions per episode. A uniform-random policy dies after about 180 ticks with **8 px** of progress (measured on `train_extra` 3000-3049): it never reaches the first wall. |
| Moving left at spawn is lethal | [`Game.X_MIN`](../infinite_maze/core/game.py) equals `PLAYER_START_X` (80), so one LEFT tick at spawn terminates ([`Simulation.step`](../infinite_maze/core/simulation.py)). Every run's first 45k steps have mean episode length 46-58 decisions. Early training mostly teaches "never press left", the move needed to back out of dead ends. |
| Turning needs pixel alignment | A 10 px player must sit inside a 22 px gap to turn; at 4 px per decision the alignment window is about 3 decisions wide. |
| The agents did not stop exploring; they dithered | Policy entropy stayed at 1.0-1.3 nats (max 1.61) in every run (`metrics.csv`). Undirected pixel dithering does not cover multi-cell detours. |
| The "information ceilings" assumed perfect control | The oracle and `WindowedPlanner` include a hand-coded 2 px lane keeper ([`OraclePolicy.act_privileged`](../training/oracle.py)). 839 / 1327 measure planning with ideal control; PPO had to learn both together. |
| Cloning failed for the same reason | Plan 4 section 15: closed-loop pixel errors compound. |
| "Network design ruled out" overstated | One CNN tried: [`MazeCNN`](../training/features.py) has a 9-raster-cell (4.5 maze cell) receptive field before a single linear layer. The egocentric 11 px raster also shifts with the player's pixel offset (aliasing). |
| Slow, single-seed iteration | 1-4.4 h per run, seed 0 only, evaluation noise about +/-30, about 25 h of compute for seven single-knob comparisons. |

Even at cell level, exploration is not free: a uniform random walk over cells for 136 moves (one 3000-tick episode) gains 1.2 columns and dies in 47/50 validation mazes (the spawn lane is next to the kill line); with lethal-left masked it gains 3.8 columns (max 8). That is why this plan pairs the cell interface with exploration levers.

## 2. Decisions adopted

| # | Decision | Choice | Reason |
|---|---|---|---|
| D19 | Action interface | **Cell-level macro actions** (stay, cell right / left / up / down), executed by a scripted lane keeper that reads only the walls next to the player. Pixel-level PPO is paused (D24) | Removes the control problem so navigation can be measured on its own; about 136 decisions per episode instead of 750 |
| D20 | Environment | A macro-action wrapper over the **real `Simulation`** (ticks, pace, scroll, death, recycling all unchanged). No separate cell-graph simulator unless measured throughput is below the threshold in C4 | Single source of truth: no rule duplication, no fidelity drift; training, evaluation and replays use the same ticks |
| D21 | Observation | **Screen-aligned cell grid**: the maze cells currently on screen (kill line to the right screen edge, all 19 rows), channels `open_right`, `open_down`, `player`; plus scalars (kill-line sub-cell offset, pace, pace phase). L1 adds a `visited` channel | Exactly what a human sees, so D16 holds by construction; no pixel aliasing; small enough for a deep CNN |
| D22 | Oracle use | Allowed only to choose **reset states** in the L2 curriculum (run the oracle for a random prefix, then hand over). Never in observations, rewards or evaluation. Evaluation always starts at the real spawn | Extends D17 for one purpose: putting the agent where detours happen. The learner is still judged from spawn |
| D23 | Evidence standard | Every reported result is the **median of 3 seeds** (`seed` 0, 1, 2); single-seed numbers are labelled as such | Evaluation noise is about +/-30; the cell runs are cheap enough for this |
| D24 | Stop rule | No new pixel-level PPO run until gate C passes or this plan is closed | Pixel runs cannot answer the navigation question |
| D25 | Fixed hyper-parameters for Phase B | One PPO setting for every lever (section 3.5); no tuning between levers | One change at a time |
| D26 | Unchanged | Reward (`v1`, unshaped), `max_steps` 3000 ticks, `start_pace` 0, seed sets, ladder thresholds, evaluation reward, `frame_skip` 4 inside the executor | Tables stay comparable with v1-v5a |
| D27 | Who runs what | `ml-dev` may launch runs under 30 minutes (smoke, gate A, Phase B lever runs); longer runs are launched by the user | The fast loop is the point of this plan |

Unchanged from earlier plans: D11 (shaping shelved), D16 (no privileged observations), D18 (`train_extra` for training), `test` seeds untouched until the final report.

## 3. Design

```text
            ┌────────────── training (CellMazeEnv) ───────────────┐
cell action │  CellMacroExecutor ── pixel action per 4 ticks ──►   │  repeat_action ─► Simulation.step
 (0..4)  ──►│  (lane keeping, stop at next cell centre)            │      (unchanged rules, reward per tick)
            │  build_cell_observation(sim) ◄───────────────────────┘
            └──────────────────────────────────────────────────────┘
evaluation:  CellModelPolicy (Policy adapter) = network + same executor, through evaluate() / expert / overlay
```

### 3.1 `training/cell_actions.py` — `CellMacroExecutor`

```python
CELL_ACTIONS = ("stay", "right", "left", "up", "down")  # indices match MOVEMENT_CONSTANTS

class CellMacroExecutor:
    def __init__(self, frame_skip: int = 4, max_ticks: int = CELL_MACRO_MAX_TICKS) -> None: ...
    def start(self, sim: Simulation, cell_action: int) -> None: ...   # picks the target cell
    def next_pixel_action(self, sim: Simulation) -> Optional[int]: ...  # None when the macro is done
```

- Target: the neighbouring cell's centre in world coordinates (`MazeGeometry` cell lattice, reloaded per episode). Lane keeping reuses the oracle's alignment rule (align the cross axis within `ORACLE_ALIGN_TOL_PX`, then move along the main axis), factored into a shared helper so the oracle's behaviour and tests do not change.
- **Blocked move** (wall between cells, per `MazeGeometry.passable`) and `stay`: issue `DO_NOTHING` for `CELL_STAY_TICKS` (= `MAZE_CELL_SIZE`, 22) ticks. Same time cost as a move, so blocked actions are not a free way to burn decisions.
- Terminates early on death or truncation; stops at `max_ticks` (safety cap, about 2 cells' worth) and records it as a stuck macro in `info`.
- Reads only the walls bordering the current and target cell (on screen). A test asserts the executor's pixel actions are identical for two simulations whose walls differ only away from those cells.

### 3.2 `training/cell_observation.py`

```python
@dataclass
class CellObservation:
    grid: List[List[List[int]]]   # [channel][row][col], rows = 19, cols = CELL_OBS_COLS
    scalars: Tuple[float, ...]    # kill_line_offset, pace, pace_phase

def build_cell_observation(sim, geometry, visited: Optional[Set[Cell]] = None) -> CellObservation: ...
def cell_observation_size(visited: bool) -> int: ...
```

- Column 0 is the maze column containing the kill line (screen x `X_MIN`); `CELL_OBS_COLS` = `ceil((SCREEN_WIDTH - X_MIN) / MAZE_CELL_SIZE) + 1` = 27 covers to the right screen edge. All 19 rows are on screen.
- Channels: `open_right[r][c]` and `open_down[r][c]` from `MazeGeometry.passable`; `player` one-hot at the player's cell; optional `visited` (cells entered this episode, maintained by the env and by `CellModelPolicy` through one shared tracker).
- Note on the kill line: the scroll offset grows from pace **and** from the `X_MAX` clamp, so the kill line trails the player's furthest progress by at most 240 px (about 11 cells). Backing out further than that is lethal; column 0 of the grid shows it.
- D16 test, same pattern as plan 4: identical observation for two simulations whose walls differ only off screen; no field derived from distances, the oracle or the potential.

### 3.3 `training/cell_env.py` — `CellMazeEnv`

- Same constructor shape as [`MazeEnv`](../training/env.py) (`seeds`, `start_pace`, `max_steps`, `frame_skip`, `reward_config`, `reward_scale`) plus `visited: bool`, `curriculum: CellCurriculumConfig`, `count_bonus: float`.
- `step(cell_action)`: runs the executor through `repeat_action` until the macro ends; reward is the sum of the per-tick `RewardFunction` totals (identical reward code to every earlier run) times `reward_scale`, plus the training-only count bonus (L3). `info` adds `base_reward`, `bonus`, `macro_ticks`, `max_column`, `death_cause` (`"left_edge"` or `"scroll"`), `stuck_macro`.
- `reset()`: with probability `curriculum.prefix_prob` (L2) run the oracle through the same executor for `T ~ U[0, curriculum.max_prefix_ticks]` ticks, then hand over (the episode keeps its tick count, so `max_steps` still caps it). Otherwise start at spawn.
- `check_env` passes; with `prefix_prob` 0, `count_bonus` 0, the summed reward of an episode equals the `evaluate()` return of the same cell-action sequence through `CellModelPolicy`.

### 3.4 Config, policy and harness wiring

- `ExperimentConfig.action_space: "pixel" | "cell"` (default `"pixel"`; every old config loads unchanged). New sections, unknown keys rejected, defaults off: `cell_observation: {visited: false}`, `cell_curriculum: {prefix_prob: 0.0, max_prefix_ticks: 2000}`, `exploration: {count_bonus: 0.0}`. Cell-only fields are rejected when `action_space` is `"pixel"`.
- `ppo.features`: add `"cell_cnn"` (L0) and `"cell_resnet"` (L4), in [features.py](../training/features.py).
  - `cell_cnn`: four 3x3 convolutions, stride 1, 32 channels, then a linear layer of 256 with the scalars concatenated. Receptive field 9 cells.
  - `cell_resnet`: IMPALA-style, 3 stages (16/32/32 channels, two residual blocks each, max-pool stride 2 in stages 2-3), linear 256.
- `CellModelPolicy(Policy)`: network plus executor plus visited tracker; implements `act_privileged(sim)` only so it can drive the executor (it reads `sim` only through `build_cell_observation` and the executor, both D16-tested). `build_policy`, `evaluate` CLI, `training.expert`, `train.py` evaluation and replays, and `metadata.json` handle `action_space: "cell"`. Replays stay tick-exact because the executor goes through `repeat_action` with the recorder.
- Model loading refuses a cell model in a pixel setup and vice versa, with a clear error.

### 3.5 Phase B PPO settings (D25)

`n_envs` 6, `n_steps` 256, `batch_size` 256, `n_epochs` 10, `learning_rate` 3e-4, `gamma` 0.99 (one decision is now about 22 ticks; 0.99 keeps a horizon of about 100 cells), `gae_lambda` 0.95, `ent_coef` 0.01, `clip_range` 0.2, `net_arch` [64, 64] heads, `train_reward_scale` 0.05, `total_timesteps` 2,000,000 cell decisions (about 44M ticks, the tick budget of 11M pixel decisions), `eval_every_timesteps` 200,000, `train_seeds: "train_extra"`, eval sets `train_extra` 3000-3019 and `validation`, `select_on: "validation"`.

Constants (`CELL_OBS_COLS`, `CELL_STAY_TICKS`, `CELL_MACRO_MAX_TICKS`, bonus defaults) go in [training/config.py](../training/config.py).

## 4. Work packages

```text
C1 executor ─┬─► C3 env + wiring ─► C4 gate A ─► C6 Phase B (L0 → L4) ─► C7 Phase C report ─► (C8 Phase D, decision)
C2 cell obs ─┘                         ▲
C5 stall diagnostics ──────────────────┘
```

### C1 — `CellMacroExecutor`

Section 3.1. Acceptance: unit tests on seeded mazes for each direction (ends centred in the target cell within tolerance), blocked moves cost `CELL_STAY_TICKS` and move nothing, death and truncation stop the macro, D16 locality test, oracle tests unchanged after the lane-keeping helper is factored out.

### C2 — Cell observation

Section 3.2. Acceptance: hand-checked grids on two seeds (spawn and after 1000 ticks of oracle play), the off-screen-walls D16 test, kill-line column correct after both kinds of scroll, `visited` channel off by default.

### C3 — `CellMazeEnv` and wiring

Sections 3.3-3.4. Acceptance: `check_env`; determinism (same seed and actions give identical observations and rewards); reward equality with `evaluate()` (section 3.3); old configs and `config.json` files load and behave identically; `smoke_v6.json` (cell, about 20 s) runs in the slow suite and its model plays through `evaluate()`, `training.expert` and the overlay.

### C4 — Gate A: executor fidelity, ceilings, throughput (`ml-dev` runs it)

On the 50 validation seeds (`max_steps` 3000, `start_pace` 0, `frame_skip` 4), through `CellModelPolicy`-style adapters:

| Policy | Purpose |
|---|---|
| Oracle choosing cells (`next_cell`) through the executor | Fidelity: must score **>= 95% of 1478** (>= 1404), death rate <= 4% |
| `WindowedPlanner` restricted to the cell observation's columns, through the executor | Navigation ceiling for a memoryless reader of the screen |
| Random cell policy; "right, else down, else up" cell heuristic | Floors for the new interface |

Also measure CellMazeEnv decisions/s with 6 envs. **Gate A passes** if the oracle threshold holds and a 2M-decision run is estimated at 45 minutes or less. If the oracle fails, fix the executor before anything else. If throughput fails, profile first; a separate cell-graph simulator (D20) is the last resort and then needs its own fidelity test against this gate's numbers.

### C5 — Stall diagnostics

Log per-episode `max_column`, `death_cause` and `stuck_macro` counts into `metrics.csv` (rollout means) and per-set means into `eval.json`, for both action spaces. Old runs' files stay readable.

### C6 — Phase B: learnability ladder (3 seeds per rung, `ml-dev` runs)

Configs `training/configs/v6_<lever>_s<seed>.json`, each rung adds exactly one lever to the previous rung:

| Rung | Config stem | Adds |
|---|---|---|
| L0 | `v6_l0_cell` | Cell actions + cell observation + `cell_cnn`, nothing else |
| L1 | `v6_l1_visited` | `visited` channel |
| L2 | `v6_l2_prefix` | Oracle-prefix reset curriculum, `prefix_prob` 0.5, `max_prefix_ticks` 2000 |
| L3 | `v6_l3_count` | Episodic count bonus `count_bonus / sqrt(n(cell))` on entering a cell, `count_bonus` 0.5 (scaled units; one new cell pays about half its progress reward) |
| L4 | `v6_l4_resnet` | `cell_resnet` features |

Run L0 first and stop at the first rung that passes gate B. After a pass, ablate backwards (drop each earlier lever once, 3 seeds) to find which levers matter, then carry only those forward.

**Gate B** (criteria fixed now; validation, real game through `evaluate()`, median of 3 seeds, `best` selected on validation):

| Result | Reading | Next |
|---|---|---|
| Median score >= 740 (rung 2: 0.5 x oracle) | Navigation is learnable; control was the blocker | C7 |
| 500 <= median < 740 | Learnable, not yet expert | C7, plus one longer run (10M decisions) of the passing rung, user-launched |
| 250 < median < 500 after L4 | Partial | Longer run of the best rung before concluding; then C7 |
| Median <= 250 after L4 | PPO from scratch cannot learn navigation here even without the control problem | Close this plan; next candidates are a recurrent policy (`sb3-contrib`) or a different algorithm, written up as plan 6 |

Also report per rung: `train_extra` score (memorisation check), `max_column`, death causes, and the gap to the windowed ceiling from C4.

### C7 — Phase C: back in the real game, report

- Already measured on the real `Simulation` by construction (D20). **Gate C**: the passing model's `evaluate()` score matches its in-training evaluation within noise, and its replays show no executor pathologies (stuck macros under 1% of macros).
- Expert ladder for training and validation sets (`python -m training.expert --run ... --model best`).
- Independence check (plan 4 section 10.4): `start_pace` 1, short horizon, compared with the oracle and the windowed ceiling.
- Watch best and worst replays in the overlay and describe stalls, oscillation and exploits.
- `training/README.md` results section in the v1-v5a table style, stating which gate B row was hit.

### C8 — Phase D: fully learned control (decision after C7)

Only if the owner wants the agent to learn pixel control as well. Options, to be decided then:

1. Learn the low-level controller with RL ("reach the adjacent cell centre": short horizon, dense reward), replacing the scripted executor.
2. Distil the hierarchical agent into a flat pixel policy with DAgger. The teacher is now a non-privileged policy that sees the same screen, which removes the label inconsistency that sank the plan 4 probes.

Until then the deliverable is a hierarchical agent: learned navigation, scripted lane keeping. Report it as such.

## 5. Success criteria

| Outcome | Criterion |
|---|---|
| Minimum | Gate B row ">= 500", that is rung 1 on validation with a clear margin over the heuristic (200), death rate <= 10% |
| Target | Median validation score >= 740 (ladder rung 2) |
| Conclusive negative | Gate B row "<= 250 after L4", written up with the numbers; this rules out the action interface as the remaining cause |

Either a pass or the conclusive negative is a successful end to this plan.

## 6. Risks

| Risk | Mitigation |
|---|---|
| The executor is a hidden planner | It only reaches an adjacent cell with walls it can see next to the player; D16 locality test; gate A floors (random and heuristic cell policies) show what the executor alone is worth |
| Oracle-prefix resets leak privileged information | Used only to choose start states (D22); evaluation from spawn only; a test that `CellMazeEnv` with `prefix_prob` 0 never calls the oracle |
| Count bonus rewards wandering instead of progress | Bonus is per new cell, decays with visits, training only; watch `max_column` and replays; drop it if the ablation shows no gain |
| Lethal left at spawn still dominates early learning | Visible in `death_cause`; if more than half of early deaths are `left_edge` after 200k decisions at L0, flag it (masking it would change the game, so it is a decision, not a fix) |
| Throughput is lower than estimated | Gate A measures it before Phase B; profile before building a separate simulator |
| Gate B passes for a reason we cannot name | Backward ablation is mandatory before C7 |
| Comparing cell and pixel tables | Same evaluation reward, horizon, pace, seeds and `evaluate()` harness; the action interface is stated in every table |

## 7. Definition of done

- C1-C5 delivered with tests; CI parity commands green (`black`, `flake8`, `mypy` on `infinite_maze/ training/`, `pytest -m "not slow and not performance"`); core install unaffected; `infinite_maze/` unchanged.
- Gate A and gate B outcomes recorded in `training/README.md` with numbers, including negative results.
- If gate B passes: C7 complete, with the ladder, the independence check and the replay review.
- If gate B fails conclusively: plan 6 drafted from the evidence.

## 8. Out of scope

Pace curriculum, reward shaping (D11), changing game rules (lethal left, scroll speed), recurrent policies (unless gate B fails), and any use of the `test` seeds.
