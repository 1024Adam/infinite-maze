"""Experiment configs: JSON files that pin everything about a training run.

Stdlib only, so parsing and validation are testable without the ML groups.
Unknown keys are rejected everywhere, and the ``test`` seed set can never be
used for training or selection.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

from . import config
from .reward import RewardConfig
from .shaping import ShapingConfig

SeedSpec = Union[str, Sequence[int]]


def resolve_seeds(spec: SeedSpec) -> List[int]:
    """Turn a seed-set name (``"validation"``) or an explicit list into seeds."""
    if isinstance(spec, str):
        if spec not in config.SEED_SETS:
            raise ValueError(
                f"unknown seed set {spec!r}; choose from {sorted(config.SEED_SETS)}"
            )
        seeds = list(config.SEED_SETS[spec])
    else:
        if isinstance(spec, (bytes, dict)) or not all(
            isinstance(s, int) and not isinstance(s, bool) for s in spec
        ):
            raise ValueError(f"seeds must be a set name or a list of ints: {spec!r}")
        seeds = list(spec)
    if not seeds:
        raise ValueError("a seed set must not be empty")
    forbidden = set(config.TEST_SEEDS)
    if forbidden.intersection(seeds):
        raise ValueError("the 'test' seeds must not be used in a training config")
    return seeds


@dataclass(frozen=True)
class CurriculumStage:
    """From ``timesteps`` onward, train with maze ``start_pace``."""

    timesteps: int
    start_pace: int


@dataclass(frozen=True)
class ObservationConfig:
    """Size of the egocentric wall raster (the rest of the observation is fixed)."""

    grid_size: int = config.OBS_GRID_SIZE
    cell_px: int = config.OBS_GRID_CELL_PX

    def validate(self) -> None:
        """Reject rasters that are empty or look beyond the rendered screen."""
        if self.grid_size < 1 or self.cell_px < 1:
            raise ValueError("observation grid_size and cell_px must be at least 1")
        if self.grid_size * self.cell_px > config.OBS_MAX_WINDOW_PX:
            raise ValueError(
                f"observation window {self.grid_size * self.cell_px} px exceeds the "
                f"{config.OBS_MAX_WINDOW_PX} px screen: the agent may only see what "
                f"is rendered"
            )


@dataclass
class PPOConfig:
    """PPO hyper-parameters (starting points, not tuned values)."""

    policy: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    n_steps: int = 512
    batch_size: int = 256
    learning_rate: float = 0.0003
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    n_epochs: int = 10
    clip_range: float = 0.2


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce a run (copied into the run directory)."""

    name: str
    train_seeds: SeedSpec
    total_timesteps: int
    algo: str = "ppo"
    eval_seeds: Dict[str, SeedSpec] = field(default_factory=dict)
    select_on: str = ""  # eval set whose mean return picks best.zip
    start_pace: int = config.EVAL_START_PACE
    curriculum: List[CurriculumStage] = field(default_factory=list)
    max_steps: int = config.EVAL_MAX_STEPS
    frame_skip: int = config.DEFAULT_FRAME_SKIP
    reward: RewardConfig = field(default_factory=RewardConfig)
    shaping: ShapingConfig = field(default_factory=ShapingConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    train_reward_scale: float = 1.0  # training only; evaluation stays unscaled
    n_envs: int = 1
    ppo: PPOConfig = field(default_factory=PPOConfig)
    eval_every_timesteps: int = 100_000
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.name or any(c in self.name for c in "/\\"):
            raise ValueError("name must be a non-empty single path component")
        if self.algo != "ppo":
            raise ValueError(f"unsupported algo {self.algo!r}; only 'ppo' exists")
        for label, value in (
            ("total_timesteps", self.total_timesteps),
            ("max_steps", self.max_steps),
            ("frame_skip", self.frame_skip),
            ("n_envs", self.n_envs),
            ("eval_every_timesteps", self.eval_every_timesteps),
        ):
            if value < 1:
                raise ValueError(f"{label} must be at least 1")
        if self.start_pace < 0:
            raise ValueError("start_pace must not be negative")
        self.reward.validate(self.max_steps)
        self.observation.validate()
        if self.shaping.weight < 0:
            raise ValueError("shaping.weight must not be negative")
        if self.train_reward_scale <= 0:
            raise ValueError("train_reward_scale must be positive")
        # Only enforced when shaping is on, so older configs keep loading.
        if self.shaping.weight > 0 and self.shaping.gamma != self.ppo.gamma:
            raise ValueError(
                f"shaping.gamma ({self.shaping.gamma}) must equal "
                f"ppo.gamma ({self.ppo.gamma})"
            )
        resolve_seeds(self.train_seeds)
        for seeds in self.eval_seeds.values():
            resolve_seeds(seeds)
        if self.eval_seeds and self.select_on not in self.eval_seeds:
            raise ValueError(
                f"select_on {self.select_on!r} must be one of {sorted(self.eval_seeds)}"
            )
        stages = [stage.timesteps for stage in self.curriculum]
        if stages != sorted(set(stages)) or any(t < 0 for t in stages):
            raise ValueError("curriculum timesteps must be unique and increasing")

    def train_seed_list(self) -> List[int]:
        """Maze seeds the training environments draw from."""
        return resolve_seeds(self.train_seeds)

    def eval_seed_lists(self) -> Dict[str, List[int]]:
        """Evaluation seed lists by label (training and validation, kept apart)."""
        return {label: resolve_seeds(s) for label, s in self.eval_seeds.items()}

    def pace_at(self, timesteps: int) -> int:
        """Training ``start_pace`` in effect at ``timesteps``."""
        pace = self.start_pace
        for stage in self.curriculum:
            if timesteps >= stage.timesteps:
                pace = stage.start_pace
        return pace

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly form (round-trips through ``from_dict``)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        """Build from parsed JSON; unknown keys at any level are an error."""
        _reject_unknown("config", values, {f.name for f in fields(cls)})
        data = dict(values)
        if "reward" in data:
            _reject_unknown("reward", data["reward"], _names(RewardConfig))
            data["reward"] = RewardConfig(**data["reward"])
        if "observation" in data:
            _reject_unknown(
                "observation", data["observation"], _names(ObservationConfig)
            )
            data["observation"] = ObservationConfig(**data["observation"])
        if "shaping" in data:
            _reject_unknown("shaping", data["shaping"], _names(ShapingConfig))
            data["shaping"] = ShapingConfig(**data["shaping"])
        if "ppo" in data:
            _reject_unknown("ppo", data["ppo"], _names(PPOConfig))
            data["ppo"] = PPOConfig(**data["ppo"])
        if "curriculum" in data:
            stages = []
            for stage in data["curriculum"]:
                _reject_unknown("curriculum stage", stage, _names(CurriculumStage))
                stages.append(CurriculumStage(**stage))
            data["curriculum"] = stages
        return cls(**data)


def _names(cls: type) -> set:
    return {f.name for f in fields(cls)}


def _reject_unknown(where: str, values: Any, allowed: set) -> None:
    if not isinstance(values, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {where} keys {sorted(unknown)}")


def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load and validate an experiment config from a JSON file."""
    with open(path) as handle:
        return ExperimentConfig.from_dict(json.load(handle))
