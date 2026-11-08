"""Action repeat shared by the environment and the evaluation harness."""

from typing import NamedTuple, Optional

from infinite_maze.core.replay import ReplayRecorder
from infinite_maze.core.simulation import Simulation
from infinite_maze.utils.config import config as game_config

from .reward import RewardFunction


class RepeatResult(NamedTuple):
    """Outcome of repeating one action for up to ``frame_skip`` ticks."""

    reward: float
    terminated: bool
    truncated: bool
    ticks: int  # simulation ticks actually advanced (early stop on episode end)


def repeat_action(
    sim: Simulation,
    action: int,
    reward: RewardFunction,
    frame_skip: int,
    recorder: Optional[ReplayRecorder] = None,
) -> RepeatResult:
    """Apply ``action`` for ``frame_skip`` ticks, summing the reward.

    Stops early when the episode terminates or is truncated, so ``max_steps`` keeps
    its meaning of simulation ticks. Every tick is recorded when a ``recorder`` is
    given, so replays are tick-exact whatever the action repeat.
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    intent = game_config.ACTION_TO_INTENT[action]
    total = 0.0
    for tick in range(1, frame_skip + 1):
        result = sim.step(intent)
        if recorder is not None:
            recorder.record(intent)
        total += reward(result).total
        if result.terminated or result.truncated:
            return RepeatResult(total, result.terminated, result.truncated, tick)
    return RepeatResult(total, False, False, frame_skip)
