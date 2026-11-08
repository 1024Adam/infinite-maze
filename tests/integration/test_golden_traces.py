"""
Golden-trace characterization tests for the game loops.

Each scenario in ``tests/fixtures/golden_scenarios.py`` drives the real engine
loops with scripted input and compares the per-frame state to a committed trace
in ``tests/fixtures/golden/``.

Regenerate ONLY after an intentional gameplay change, and explain every diff:

    REGENERATE_GOLDENS=1 poetry run pytest tests/integration/test_golden_traces.py
"""

import os

import pytest

from tests.fixtures.golden_harness import loadGolden, saveGolden
from tests.fixtures.golden_scenarios import SCENARIOS


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_trace(name):
    """The scenario's per-frame trace matches the committed golden file."""
    description, run = SCENARIOS[name]
    trace = run()

    if os.environ.get("REGENERATE_GOLDENS"):
        saveGolden(name, description, trace)
        return

    assert trace == loadGolden(name)


def test_goldens_exercise_interesting_behaviour():
    """Guard against goldens that silently stop covering what they claim."""
    if os.environ.get("REGENERATE_GOLDENS"):
        pytest.skip("regenerating")
    traces = {name: loadGolden(name) for name in sorted(SCENARIOS)}

    def frames(name):
        """Frame entries (without control/end markers) of a golden trace."""
        return [e for e in traces[name] if e["label"] == "frame"]

    blocked = frames("walk_right_until_blocked")
    assert blocked[-1]["x"] == blocked[-20]["x"] < 320

    assert max(e["x"] for e in frames("clamp_at_x_max")) == 320
    assert len({e["lines"] for e in frames("fuzz_pace1_recycle")}) > 1
    assert traces["left_score_floor"][-1]["score"] == 0
    assert traces["idle_death_pace1"][-1]["label"] == "end"
    assert traces["controlled_idle_death_pace1"][-1]["label"] == "gameover"
    for name in ("fuzz_pace2", "fuzz_pace3"):
        assert {e["pace"] for e in frames(name)} == {int(name[-1])}
