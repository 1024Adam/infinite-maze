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

# Reward
REWARD_PROGRESS_WEIGHT: float = 1.0  # per pixel moved right (negative if left)
REWARD_STEP_PENALTY: float = 0.01  # per step
REWARD_DEATH_PENALTY: float = 100.0  # on terminated
REWARD_TRUNCATION_BONUS: float = 0.0  # on truncated (survived the cap)

# Evaluation
TRAIN_SEEDS = range(0, 100)
VALIDATION_SEEDS = range(1000, 1050)
TEST_SEEDS = range(2000, 2050)
SEED_SETS = {
    "train": TRAIN_SEEDS,
    "validation": VALIDATION_SEEDS,
    "test": TEST_SEEDS,
}
EVAL_MAX_STEPS: int = 2000
EVAL_START_PACE: int = 0
DEFAULT_FRAME_SKIP: int = 4

# Oracle
ORACLE_LOOKAHEAD_COLS: int = 12  # BFS target: this many columns ahead of the player
ORACLE_ALIGN_TOL_PX: int = 2  # lane-keeping tolerance when centring in a corridor
ORACLE_CACHE_FIELDS: int = 64  # BFS distance fields kept per episode
ORACLE_STATS_MAX_PACE: int = 6  # oracle_stats sweeps start_pace 0..this
TORCH_THREADS: int = 1  # keeps CPU PPO reproducible and leaves cores to the envs

# Expert ladder thresholds (rungs 2-4). Provisional until the oracle ceilings
# (python -m training.oracle_stats) say what is achievable.
EXPERT_LADDER = {
    "r2": 0.5,  # rung 2: mean score >= r2 * oracle mean score
    "r3": 0.9,  # rung 3: mean score >= r3 * oracle mean score
    "pace": 3,  # rung 4: mean final pace >= pace ...
    "max_death_rate": 0.1,  # ... with death rate <= this
}
RUNS_DIR = Path(__file__).parent / "runs"

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
