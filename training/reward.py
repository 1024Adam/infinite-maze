"""Reward signal, kept separate from the in-game score.

The in-game score is floored at 0 and only moves by +/-1. The training reward is
based on the actual horizontal displacement reported by ``StepResult.moved``, so
it is neither floored nor affected by scrolling.
"""

import json
from dataclasses import asdict, dataclass
from typing import Optional

from infinite_maze.core.simulation import StepResult

from . import config


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the reward terms."""

    progress_weight: float = config.REWARD_PROGRESS_WEIGHT  # per pixel moved right
    step_penalty: float = config.REWARD_STEP_PENALTY  # per step
    death_penalty: float = config.REWARD_DEATH_PENALTY  # on terminated
    truncation_bonus: float = config.REWARD_TRUNCATION_BONUS  # on truncated

    def validate(self, max_steps: Optional[int]) -> None:
        """Reject configs where dying could beat surviving the whole episode."""
        if max_steps is not None and self.death_penalty <= (
            self.step_penalty * max_steps
        ):
            raise ValueError(
                f"death_penalty ({self.death_penalty}) must exceed the step "
                f"penalty accrued over max_steps "
                f"({self.step_penalty} * {max_steps} = "
                f"{self.step_penalty * max_steps}); otherwise dying early is "
                f"better than surviving"
            )


@dataclass
class RewardBreakdown:
    """Where one step's reward came from."""

    progress: float
    step: float
    terminal: float
    total: float


class RewardFunction:
    """Computes the reward of a step from its ``StepResult``."""

    def __init__(self, reward_config: Optional[RewardConfig] = None) -> None:
        self.config = reward_config if reward_config is not None else RewardConfig()

    def __call__(self, result: StepResult) -> RewardBreakdown:
        progress = self.config.progress_weight * result.moved[0]
        step = -self.config.step_penalty
        terminal = 0.0
        if result.terminated:
            terminal -= self.config.death_penalty
        if result.truncated:
            terminal += self.config.truncation_bonus
        return RewardBreakdown(progress, step, terminal, progress + step + terminal)


def load_reward_config(path: str) -> RewardConfig:
    """Load a ``RewardConfig`` from a JSON object; unknown keys are an error."""
    with open(path) as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"{path}: reward config must be a JSON object")
    unknown = set(values) - set(asdict(RewardConfig()))
    if unknown:
        raise ValueError(f"{path}: unknown reward config keys {sorted(unknown)}")
    return RewardConfig(**values)
