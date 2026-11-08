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
RUNS_DIR = Path(__file__).parent / "runs"

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
