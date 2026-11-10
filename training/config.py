"""Constants for observations, rewards, baselines and evaluation.

The seed sets are the permanent benchmark: define them once and never change
them silently. Do not tune on ``TEST_SEEDS``.
"""

from pathlib import Path

# Observation: egocentric wall raster plus rays
OBS_GRID_SIZE: int = 9  # raster is OBS_GRID_SIZE x OBS_GRID_SIZE cells
OBS_GRID_CELL_PX: int = 11  # 9 * 11 = 99 px, about 4.5 maze cells
OBS_RAY_CAP_PX: int = 110  # rays report at most this distance
OBS_PACE_CAP: int = 10  # pace is normalised against this value
OBS_MAX_WINDOW_PX: int = 640  # a raster may not look beyond the rendered screen width

# Reward
REWARD_PROGRESS_WEIGHT: float = 1.0  # per pixel moved right (negative if left)
REWARD_STEP_PENALTY: float = 0.01  # per step
REWARD_DEATH_PENALTY: float = 100.0  # on terminated
REWARD_TRUNCATION_BONUS: float = 0.0  # on truncated (survived the cap)

# Evaluation
TRAIN_SEEDS = range(0, 100)
VALIDATION_SEEDS = range(1000, 1050)
TEST_SEEDS = range(2000, 2050)
# More training mazes (plan 4 D18), added after the sweep showed that 80-100 mazes
# are memorised. Training and probe fitting only: never evaluate "validation"
# claims on it. The sets above are unchanged.
TRAIN_EXTRA_SEEDS = range(3000, 4000)
SEED_SETS = {
    "train": TRAIN_SEEDS,
    "validation": VALIDATION_SEEDS,
    "test": TEST_SEEDS,
    "train_extra": TRAIN_EXTRA_SEEDS,
}
EVAL_MAX_STEPS: int = 2000
EVAL_MAX_STEPS_PROBE: int = 3000  # the horizon of the v1/v2 tables and ladder
EVAL_START_PACE: int = 0
DEFAULT_FRAME_SKIP: int = 4

# Oracle
ORACLE_LOOKAHEAD_COLS: int = 12  # BFS target: this many columns ahead of the player
ORACLE_ALIGN_TOL_PX: int = 2  # lane-keeping tolerance when centring in a corridor
ORACLE_CACHE_FIELDS: int = 64  # BFS distance fields kept per episode
ORACLE_STATS_MAX_PACE: int = 6  # oracle_stats sweeps start_pace 0..this
TORCH_THREADS: int = 1  # keeps CPU PPO reproducible and leaves cores to the envs

# Expert ladder thresholds (rungs 2-4), set from the oracle ceilings at
# start_pace 0, max_steps 3000 (oracle mean score about 1,475, heuristic about 190,
# oracle final pace 1, so rung 4 can only ask for pace 1 at that horizon).
EXPERT_LADDER = {
    "r2": 0.5,  # rung 2: mean score >= r2 * oracle mean score
    "r3": 0.9,  # rung 3: mean score >= r3 * oracle mean score
    "pace": 1,  # rung 4: mean final pace >= pace ...
    "max_death_rate": 0.1,  # ... with death rate <= this
}
RUNS_DIR = Path(__file__).parent / "runs"

# Observation probe (plan 4 section 10): behaviour cloning + DAgger on oracle labels
# decides which observation can solve the mazes before any new PPO run. It is fitted
# on train_extra mazes only (the 80 mazes of train 20-99 were memorised in the first
# sweep). Train seeds 0-19 are held out for action accuracy, and the validation seeds
# are only ever played, never labelled. The test seeds are not used.
PROBE_FIT_SEEDS = range(3000, 3400)
PROBE_HOLDOUT_SEEDS = range(0, 20)
PROBE_FIT_EVAL_SEEDS = range(3000, 3020)  # fitted mazes played for the "fit" score
PROBE_HIDDEN = (64, 64)  # same shape as the PPO policy
PROBE_BETAS = (1.0, 0.3, 0.1, 0.0)  # P(oracle acts) per round; round 0 is plain cloning
PROBE_EPOCHS: int = 30  # per round, warm-started from the previous round
PROBE_BATCH: int = 512
PROBE_LEARNING_RATE: float = 0.001
PROBE_STRIDE: int = 4  # keep every n-th decision (consecutive states are near-copies)
PROBE_WORKERS: int = 16
PROBE_FIT_THREADS: int = 8  # torch threads while fitting (rollout workers are idle)
PROBE_WINDOWS = (9, 15, 21, 31)  # raster sizes swept; 31 * 11 px = 341 px, on screen
PROBE_CONTROL_HIDDEN = (256, 256)  # capacity control for the widest window
PROBE_G1_PASS_SCORE: float = 600.0  # a window reaching this holds the answer (~40%)
PROBE_G1_FAIL_SCORE: float = 250.0  # every window at or below this: memory needed

# Benchmark
BENCH_SEEDS: int = 10
BENCH_STEPS: int = 10_000
BENCH_START_PACE: int = 0
BENCH_PROFILE_ROWS: int = 15
# Mostly right so the world scrolls like real play; the rest wanders vertically.
BENCH_ACTION_POOL = (1, 1, 1, 1, 1, 1, 3, 4, 0)

# Heuristic baseline
HEURISTIC_OPEN_PX: int = 2  # a ray longer than this counts as open
HEURISTIC_COMMIT_STEPS: int = 40  # max steps to keep a chosen vertical direction

# Overlay
OVERLAY_FONT_SIZE: int = 14
OVERLAY_PANEL_WIDTH_PX: int = 230
OVERLAY_PANEL_MARGIN_PX: int = 6
OVERLAY_LINE_HEIGHT_PX: int = 16
OVERLAY_CELL_ALPHA: int = 110
OVERLAY_PANEL_ALPHA: int = 170
