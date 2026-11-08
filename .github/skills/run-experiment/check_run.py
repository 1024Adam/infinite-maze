"""Check that a training run directory contains every expected artefact.

    python .github/skills/run-experiment/check_run.py training/runs/<name>
"""

import sys
from pathlib import Path

EXPECTED_FILES = ("config.json", "metadata.json", "metrics.csv", "eval.json")
EXPECTED_DIRS = ("checkpoints", "replays")


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    run = Path(argv[1])
    if not run.is_dir():
        print(f"missing run directory: {run}")
        return 1

    problems = [f"missing file: {name}" for name in EXPECTED_FILES if not (run / name).is_file()]
    for name in EXPECTED_DIRS:
        path = run / name
        if not path.is_dir() or not any(path.iterdir()):
            problems.append(f"missing or empty directory: {name}/")

    for problem in problems:
        print(problem)
    print("run is complete" if not problems else f"{len(problems)} problem(s) found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
