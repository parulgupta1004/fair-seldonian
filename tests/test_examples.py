"""Execute the shipped example scripts.

These are the first code a user runs, and nothing else in the suite imports
them, so a change to a public signature can break every one of them while the
tests stay green -- which is exactly what happened when ``QSA`` grew a fourth
return field. Running them here costs a little wall-clock and closes that gap.

The assertions are deliberately weak: the point is that the scripts execute
against the current API, not that their numbers hold at any particular value.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


def test_examples_are_discovered() -> None:
    """Guard against the glob silently matching nothing."""
    assert EXAMPLES, "no example scripts found - has examples/ moved?"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, (
        f"{script.name} exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    assert completed.stdout.strip(), f"{script.name} produced no output"
