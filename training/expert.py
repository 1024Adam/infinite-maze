"""Expert ladder: how close a model is to the oracle, per seed set.

    python -m training.expert --run training/runs/<name> [--model best]

``ladder()`` compares three reports that must share ``start_pace``,
``max_steps``, ``frame_skip`` and seeds. Results on fixed training seeds can be
memorised, so every ladder is reported separately for training and validation
seeds and the training one is labelled as such.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from . import config
from .evaluate import EvalReport, build_policy, evaluate
from .experiment import ExperimentConfig, load_experiment_config

RUNG_NAMES = {
    0: "below heuristic",
    1: "beats the heuristic baseline",
    2: "reaches R2 of the oracle score",
    3: "reaches R3 of the oracle score",
    4: "sustains the target pace",
}


@dataclass
class LadderResult:
    """Highest rung reached and the numbers behind each criterion."""

    rung: int
    description: str
    beats_heuristic: bool
    oracle_fraction: float
    mean_final_pace: float
    death_rate: float
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly form."""
        return asdict(self)


def _check_like_for_like(reports: Mapping[str, EvalReport]) -> None:
    keys = {
        (r.start_pace, r.max_steps, r.frame_skip, tuple(e.seed for e in r.episodes))
        for r in reports.values()
    }
    if len(keys) != 1:
        raise ValueError(
            "model, heuristic and oracle reports must use identical start_pace, "
            "max_steps, frame_skip and seeds"
        )


def ladder(
    model_report: EvalReport,
    heuristic_report: EvalReport,
    oracle_report: EvalReport,
    thresholds: Optional[Mapping[str, float]] = None,
) -> LadderResult:
    """Highest rung (0-4) the model reaches; rungs are cumulative."""
    limits = dict(config.EXPERT_LADDER if thresholds is None else thresholds)
    _check_like_for_like(
        {"model": model_report, "heuristic": heuristic_report, "oracle": oracle_report}
    )
    model = model_report.summary()
    oracleScore = oracle_report.summary()["mean_score"]
    fraction = model["mean_score"] / oracleScore if oracleScore > 0 else 0.0
    paces = [e.pace for e in model_report.episodes]
    meanPace = sum(paces) / len(paces)

    beats = model["mean_return"] > heuristic_report.summary()["mean_return"]
    reached = 0
    if beats:
        reached = 1
        if fraction >= limits["r2"]:
            reached = 2
            if fraction >= limits["r3"]:
                reached = 3
                if (
                    meanPace >= limits["pace"]
                    and model["death_rate"] <= limits["max_death_rate"]
                ):
                    reached = 4
    return LadderResult(
        rung=reached,
        description=RUNG_NAMES[reached],
        beats_heuristic=beats,
        oracle_fraction=fraction,
        mean_final_pace=meanPace,
        death_rate=model["death_rate"],
        thresholds=limits,
    )


def run_report(cfg: ExperimentConfig, model_path: str) -> Dict[str, Any]:
    """Re-run model, heuristic and oracle with the run's settings on each seed set."""
    out: Dict[str, Any] = {"model": model_path, "sets": {}}
    for label, seeds in cfg.eval_seed_lists().items():
        reports = {
            name: evaluate(
                build_policy(name, model_path),
                seeds,
                cfg.max_steps,
                cfg.start_pace,
                cfg.reward,
                cfg.frame_skip,
            )
            for name in ("random", "right", "heuristic", "oracle", "model")
        }
        result = ladder(reports["model"], reports["heuristic"], reports["oracle"])
        out["sets"][label] = {
            "memorisable": label == "train" or len(seeds) == 1,
            "ladder": result.to_dict(),
            "baselines": {n: r.summary() for n, r in reports.items()},
        }
    return out


def format_report(report: Dict[str, Any]) -> str:
    """Baseline table and ladder status per seed set, as text."""
    lines: List[str] = []
    for label, entry in report["sets"].items():
        note = " (training seeds: may be memorised)" if entry["memorisable"] else ""
        lines.append(f"== {label}{note}")
        lines.append(
            f"{'policy':<12}{'mean return':>12}{'median':>10}{'mean score':>12}"
            f"{'mean steps':>12}{'death rate':>12}"
        )
        for name, s in entry["baselines"].items():
            lines.append(
                f"{name:<12}{s['mean_return']:>12.1f}{s['median_return']:>10.1f}"
                f"{s['mean_score']:>12.1f}{s['mean_steps']:>12.0f}"
                f"{s['death_rate']:>12.0%}"
            )
        lad = entry["ladder"]
        lines.append(
            f"ladder: rung {lad['rung']} ({lad['description']}); "
            f"oracle fraction {lad['oracle_fraction']:.2f}, "
            f"mean final pace {lad['mean_final_pace']:.1f}, "
            f"death rate {lad['death_rate']:.0%}"
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: report a finished run against the baselines and ladder."""
    parser = argparse.ArgumentParser(prog="python -m training.expert")
    parser.add_argument("--run", required=True, help="run directory")
    parser.add_argument(
        "--model", default="best", help="'best', 'final' or a .zip path"
    )
    args = parser.parse_args(argv)

    run = Path(args.run)
    cfg = load_experiment_config(run / "config.json")
    model = args.model
    if model in ("best", "final"):
        model = str(run / f"{model}.zip")
    report = run_report(cfg, model)
    print(format_report(report))
    with open(run / "ladder.json", "w") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
