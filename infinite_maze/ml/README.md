# Infinite Maze — ML Training

This directory contains the reinforcement-learning pipeline for training a PPO agent to play Infinite Maze. The agent learns to navigate a procedurally generated scrolling maze while surviving increasing pace pressure.

---

## Prerequisites

The ML dependencies are declared as an optional Poetry group. Install them with:

```bash
poetry install --with ml
```

This adds `gymnasium`, `stable-baselines3`, `torch`, and `tensorboard` without affecting the base game install. `tensorboard` is only needed if you pass `--tensorboard` to `train.py`.

---

## File Overview

| File | Purpose |
|---|---|
| `environment.py` | `InfiniteMazeEnv` — gymnasium wrapper around the game engine |
| `features.py` | Observation encoding, AABB collision helpers, wall-distance scanners |
| `rewards.py` | Reward shaping functions used by each training phase |
| `train.py` | CLI script — train a PPO agent from scratch or resume from a checkpoint |
| `evaluate.py` | CLI script — score a saved model over N headless episodes |
| `diagnose.py` | CLI script — analyse action distribution to detect degenerate policies |
| `watch.py` | CLI script — watch a saved model play in a live pygame window |
| `docs/training-plan.md` | Full design document: observation space, reward table, phase schedule |

All scripts run headlessly by default (no display required). `watch.py` is the only script that opens a window.

---

## Observation Space

Each step produces a flat `float32` array of shape `(57,)` (Phase 4+) or `(54,)` (legacy models):

| Index | Feature | Range |
|---|---|---|
| 0 | Player X (normalised) | [0, 1] |
| 1 | Player Y (normalised) | [0, 1] |
| 2 | Blocked right | 0 or 1 |
| 3 | Blocked left | 0 or 1 |
| 4 | Blocked up | 0 or 1 |
| 5 | Blocked down | 0 or 1 |
| 6 | Distance to nearest wall right (normalised) | [0, 1] |
| 7 | Distance to nearest wall left (normalised) | [0, 1] |
| 8 | Distance to nearest wall up (normalised) | [0, 1] |
| 9 | Distance to nearest wall down (normalised) | [0, 1] |
| 10 | Current pace (normalised) | [0, 1] |
| 11 | Distance from death boundary (normalised) | [0, 1] |
| 12 | Consecutive ticks blocked right (normalised) | [0, 1] |
| 13 | Action repeat ratio (windowed) | [0, 1] |
| 14 | X-progress over recent window (windowed) | [0, 1] |
| 15 | Blocked-right persistence ratio (windowed) | [0, 1] |
| 16 | **Nearest right-facing gap offset** (Phase 4+) | [0, 1] |
| 17–56 | Local wall grid — 4 cols × 5 rows × 2 features | 0 or 1 |

**Gap offset feature (index 16):** Indicates the vertical direction to the nearest Y position where a RIGHT move would succeed:
- `< 0.5` → gap is **above** the player (move UP to align)
- `= 0.5` → gap is at current Y or not found within scan radius
- `> 0.5` → gap is **below** the player (move DOWN to align)

The wall grid (`obs[17..56]`) encodes the maze structure in a 4-column × 5-row window centred on the player and scanning rightward. For each cell two binary features are stored in order: **`has_right_wall`** (vertical wall on the cell's right edge) and **`has_bottom_wall`** (horizontal wall on the cell's bottom edge). Columns 0–3 span one maze cell (22 px) each, giving ~88 px of lookahead. Rows −2 to +2 are centred on the player's row.

---

## Action Space

`Discrete(5)` — maps directly to the movement constants in `config.py`:

| Integer | Action |
|---|---|
| 0 | DO_NOTHING |
| 1 | RIGHT |
| 2 | LEFT |
| 3 | UP |
| 4 | DOWN |

---

## Training Phases

Training is split into phases of increasing complexity. The `--phase` flag passed to `train.py` controls which reward-shaping function the environment uses.

| Phase | Focus | Key hyperparameters |
|---|---|---|
| 0 | Smoke test / baseline | `--timesteps 10000` |
| 1 | Basic survival, rightward bias | `--timesteps 100000` |
| 2 | Encounter pace pressure | `--timesteps 200000`, `--ent-coef 0.02` |
| 3 | Learn gap navigation (phase3_shaping active) | `--timesteps 750000`, `--n-envs 4`, `--gamma 0.995` |
| 4 | **Gap-directed navigation with explicit gap offset** | `--timesteps 200000`, `--learning-rate 1e-4`, `--clip-range 0.1` |

**Phase 4 breaking change:** Phase 4 introduces the gap offset feature (observation index 16), increasing the observation space from (56,) to (57,). Models trained before Phase 4 cannot be loaded without using `legacy=True` in evaluation scripts.

---

## train.py

Train a PPO agent. Checkpoints are saved to `checkpoints/` every 10 000 steps by default.

**Fresh training:**
```bash
python -m infinite_maze.ml.train --timesteps 100000 --phase 1
```

**Resume from a checkpoint:**
```bash
python -m infinite_maze.ml.train --timesteps 200000 --phase 2 \
  --resume checkpoints/phase1_20260330_143022/ppo_maze_final_100000_steps.zip
```

**Phase 3 — parallel envs, tighter hyperparameters:**
```bash
python -m infinite_maze.ml.train --timesteps 750000 --phase 3 \
  --resume checkpoints/phase2_20260330_160000/ppo_maze_final_200000_steps.zip \
  --n-envs 4 \
  --learning-rate 5e-5 --gamma 0.995 --n-steps 2048 \
  --batch-size 256 --n-epochs 10 --ent-coef 0.02 --clip-range 0.1
```

**Enable TensorBoard logging:**
```bash
python -m infinite_maze.ml.train --timesteps 200000 --tensorboard
tensorboard --logdir runs/
```

**Full argument reference:**

| Argument | Default | Description |
|---|---|---|
| `--timesteps` | 200 000 | Total environment steps |
| `--resume` | None | Path to a `.zip` checkpoint to continue from |
| `--phase` | 1 | Training phase (0–5); controls reward shaping |
| `--n-envs` | 1 | Parallel environments (use 4+ for Phase 3) |
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--n-steps` | 512 | Steps per rollout per env |
| `--batch-size` | 64 | Mini-batch size |
| `--n-epochs` | 10 | Gradient update epochs per rollout |
| `--ent-coef` | 0.01 | Entropy coefficient |
| `--clip-range` | 0.2 | PPO clip range |
| `--checkpoint-freq` | 10 000 | Steps between checkpoint saves |
| `--eval-freq` | 20 000 | Steps between evaluation runs (0 = off) |
| `--tensorboard` | off | Enable TensorBoard logging to `runs/` |

---

## evaluate.py

Run a saved model over N headless episodes and print per-episode scores.

```bash
python -m infinite_maze.ml.evaluate \
  --model checkpoints/ppo_maze_final_200000_steps.zip \
  --episodes 10
```

**Reproducible evaluation with a fixed seed:**
```bash
python -m infinite_maze.ml.evaluate \
  --model checkpoints/ppo_maze_final_200000_steps.zip \
  --episodes 10 --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Path to the `.zip` model file |
| `--episodes` | 10 | Number of episodes to run |
| `--phase` | 1 | Environment phase for reward-shaping context |
| `--seed` | None | Base RNG seed; episode `i` uses `seed + i` |

---

## diagnose.py

Analyse a model's action distribution to detect degenerate or learned behaviour. The key metric is **"vertical when blocked right"** — the fraction of steps where the agent chooses UP or DOWN while it cannot move right. A healthy agent uses vertical moves to find gaps; a degenerate agent spams RIGHT or idles.

```bash
python -m infinite_maze.ml.diagnose \
  --model checkpoints/ppo_maze_final_200000_steps.zip \
  --steps 5000
```

**Reproducible run:**
```bash
python -m infinite_maze.ml.diagnose \
  --model checkpoints/ppo_maze_final_200000_steps.zip \
  --steps 5000 --seed 0
```

**Interpreting the verdict:**

| Vertical-when-blocked % | Verdict |
|---|---|
| < 5 % | Degenerate — agent is not navigating the maze |
| 5 – 20 % | Emerging — some gap-seeking behaviour present |
| > 20 % | Target met — agent is genuinely navigating |

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Path to the `.zip` model file |
| `--steps` | 2000 | Number of environment steps to sample |
| `--phase` | 1 | Environment phase |
| `--seed` | None | RNG seed for reproducibility |

---

## watch.py

Watch a trained agent play in a real pygame window. Press **ESC** or close the window to stop.

```bash
python -m infinite_maze.ml.watch \
  --model checkpoints/ppo_maze_final_200000_steps.zip
```

**Watch multiple episodes at reduced speed:**
```bash
python -m infinite_maze.ml.watch \
  --model checkpoints/ppo_maze_final_200000_steps.zip \
  --episodes 5 --delay 50
```

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Path to the `.zip` model file |
| `--episodes` | 3 | Number of episodes to render |
| `--delay` | 16 | Frame delay in ms (16 ≈ 60 fps; increase to slow down) |
| `--phase` | 1 | Environment phase |

> `watch.py` uses real-time pace updates (30-second intervals) rather than the tick-based pace used during training. Behaviour may differ slightly from headless evaluation.

---

## Recommended Workflow

1. **Establish a baseline** — run a short smoke test to confirm the environment works:
   ```bash
   python -m infinite_maze.ml.train --timesteps 10000 --phase 0
   ```

2. **Train through phases sequentially**, always resuming from the previous final checkpoint.

3. **Check for degenerate behaviour after each phase** using `diagnose.py`. If vertical-when-blocked is below 5 %, the policy has collapsed — restart the phase with a higher `--ent-coef`.

4. **Visually confirm** recovered or advanced behaviour with `watch.py` before committing to a long run.

5. **Evaluate quantitively** with `evaluate.py` using a fixed seed for reproducible comparisons across checkpoints.

---

## Checkpoints

Each training run creates its own subdirectory under `checkpoints/`, named by phase and timestamp:

```
checkpoints/
  phase1_20260330_143022/
    ppo_maze_10000_steps.zip
    ppo_maze_20000_steps.zip
    ...
    ppo_maze_final_100000_steps.zip
    best/
      best_model.zip
    eval_logs/
```

The run directory is printed at the start of training so you know exactly where to find the output:
```
Run directory: checkpoints/phase1_20260330_143022
```

When resuming training, pass the full path to the checkpoint inside the relevant run directory.

---

## Running Tests

```bash
# Unit tests for the environment wrapper
pytest tests/unit/test_ml_environment.py -v

# Integration smoke test (≤ 1000 steps, CI-safe)
pytest tests/integration/test_ml_training.py -v

# All ML tests
pytest tests/unit/test_ml_environment.py tests/integration/test_ml_training.py -v
```

All tests are headless-safe — no display is required.
