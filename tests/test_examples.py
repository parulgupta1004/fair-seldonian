"""Execute the shipped example scripts.

These are the first code a user runs, and nothing else in the suite imports
them, so a change to a public signature can break every one of them while the
tests stay green -- which is exactly what happened when ``QSA`` grew a fourth
return field. Running them here costs a little wall-clock and closes that gap.

The assertions are deliberately weak: the point is that the scripts execute
against the current API, not that their numbers hold at any particular value.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


def test_examples_are_discovered() -> None:
    """Guard against the glob silently matching nothing."""
    assert EXAMPLES, "no example scripts found - has examples/ moved?"


#: Notebooks that reach the network, so they cannot run as part of the suite.
#: They are still linted for a stale API by ``test_notebooks_do_not_unpack_qsa``.
NETWORK_NOTEBOOKS = {"real_world_adult.ipynb"}

NOTEBOOKS = sorted(
    (Path(__file__).resolve().parent.parent / "examples").glob("*.ipynb")
)


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


@pytest.mark.parametrize(
    "notebook",
    [n for n in NOTEBOOKS if n.name not in NETWORK_NOTEBOOKS],
    ids=lambda p: p.name,
)
def test_notebook_executes(notebook: Path) -> None:
    """Execute the offline notebooks.

    The docs render notebooks with ``nb_execution_mode = "off"``, so a stale
    notebook is published verbatim rather than failing the build. That is worse
    than a broken script: the numbers still look authoritative. Executing them
    here keeps the committed outputs honest.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(notebook, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook.parent)}},
    )
    client.execute()  # raises CellExecutionError on any failing cell


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_do_not_unpack_qsa_as_a_three_tuple(notebook: Path) -> None:
    """Catch the stale-API break even in notebooks we cannot execute here.

    ``QSA`` returns a four-field ``QSAResult``; every three-target unpack of it
    is a ``ValueError`` waiting to happen.
    """
    source = json.loads(notebook.read_text())
    for index, cell in enumerate(source["cells"]):
        if cell["cell_type"] != "code":
            continue
        for line in "".join(cell["source"]).splitlines():
            match = re.match(r"^\s*([A-Za-z_][\w, ]*?)\s*=\s*QSA\(", line)
            if match and len(match.group(1).split(",")) == 3:
                pytest.fail(
                    f"{notebook.name} cell {index} unpacks QSA into three "
                    f"targets: {line.strip()}"
                )


def test_experiments_imports_without_matplotlib() -> None:
    """The experiments subpackage must import on a base install.

    matplotlib is an optional extra, but ``experiments/__init__`` re-exports
    ``plot_all``. When that module imported matplotlib at the top level, merely
    importing ``fair_seldonian.experiments`` -- for ``summarise``,
    ``run_study`` or ``clopper_pearson``, none of which draw anything -- raised
    ModuleNotFoundError on a base install and in CI, which installs only the
    ``dev`` extra.
    """
    script = (
        "import sys, types;"
        # Make matplotlib unimportable for this interpreter only.
        "sys.modules['matplotlib'] = None;"
        "import fair_seldonian.experiments as e;"
        "assert e.clopper_pearson(0, 40)[1] > 0.05;"
        "print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, (
        "fair_seldonian.experiments must import without matplotlib\n"
        f"--- stderr ---\n{completed.stderr}"
    )
