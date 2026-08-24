"""Execute the shipped example scripts.

These are the first code a user runs, and nothing else in the suite imports
them, so a change to a public signature can break every one of them while the
tests stay green -- which is exactly what happened when ``QSA`` grew a fourth
return field. Running them here costs a little wall-clock and closes that gap.

The assertions are deliberately weak: the point is that the scripts execute
against the current API, not that their numbers hold at any particular value.
"""

from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


def test_examples_are_discovered() -> None:
    """Guard against the glob silently matching nothing."""
    assert EXAMPLES, "no example scripts found - has examples/ moved?"


#: Notebooks that download a dataset. They run under the ``network`` marker,
#: which the default ``-m "not network"`` in pyproject excludes, so a third
#: party being down cannot fail an unrelated pull request. sklearn caches to
#: ``SCIKIT_LEARN_DATA`` when set, which keeps the download out of a home
#: directory that may not be writable.
NETWORK_NOTEBOOKS = {"real_world_adult.ipynb"}


def _notebook_params() -> list:
    return [
        pytest.param(
            n, marks=pytest.mark.network if n.name in NETWORK_NOTEBOOKS else ()
        )
        for n in NOTEBOOKS
    ]


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


@pytest.mark.parametrize("notebook", _notebook_params(), ids=lambda p: p.name)
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


# --------------------------------------------------------------------------
# Notebook output freshness
# --------------------------------------------------------------------------
#: Relative tolerance when comparing numbers inside notebook output.
#:
#: The committed outputs were produced on one machine and CI re-runs them on
#: another, so the last digits of a float can legitimately differ. This is loose
#: enough to absorb that and far tighter than any change worth noticing: the
#: staleness this exists to catch looked like 0.997 against 0.855, and a version
#: banner reading 2.1.1 against 3.1.0.
OUTPUT_RTOL = 1e-4

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _output_text(output: dict) -> str:
    """The human-readable text of one output, ignoring any binary payload."""
    if output.get("output_type") == "stream":
        return "".join(output.get("text", ""))
    return "".join(output.get("data", {}).get("text/plain", ""))


def _output_shape(cell: dict) -> list[tuple[str, str]]:
    """``(kind, text)`` per output.

    Image payloads are reduced to their mime type. Comparing PNG bytes across
    matplotlib versions and platforms would fail constantly and say nothing
    about whether the notebook is current; that a figure was produced at all is
    the part worth pinning.
    """
    shapes: list[tuple[str, str]] = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        if kind == "error":
            shapes.append(("error", output.get("ename", "")))
            continue
        if kind == "stream":
            # Jupyter splits stdout into chunks on flush boundaries, and where
            # those fall is not reproducible: the same print loop can arrive as
            # three outputs or four. Coalesce consecutive chunks from the same
            # stream, which is what a reader sees anyway.
            label = f"stream:{output.get('name', 'stdout')}"
            if shapes and shapes[-1][0] == label:
                shapes[-1] = (label, shapes[-1][1] + _output_text(output))
            else:
                shapes.append((label, _output_text(output)))
            continue
        mimes = ",".join(sorted(output.get("data", {})))
        shapes.append((f"{kind}:{mimes}", _output_text(output)))
    return shapes


def _texts_agree(committed: str, produced: str) -> bool:
    """Equal once numbers are allowed to drift within ``OUTPUT_RTOL``."""
    if committed == produced:
        return True
    # Everything that is not a number must match exactly, so a changed label,
    # a new warning or a reordered line is still a failure.
    if _NUMBER.sub("#", committed) != _NUMBER.sub("#", produced):
        return False
    for a, b in zip(_NUMBER.findall(committed), _NUMBER.findall(produced)):
        if not math.isclose(float(a), float(b), rel_tol=OUTPUT_RTOL, abs_tol=1e-9):
            return False
    return True


@pytest.mark.parametrize("notebook", _notebook_params(), ids=lambda p: p.name)
def test_notebook_outputs_are_current(notebook: Path) -> None:
    """Committed notebook outputs must match what the notebook now produces.

    Executing a notebook proves it runs; it says nothing about whether the
    outputs stored in the file are the ones it would produce today. The docs
    render notebooks with ``nb_execution_mode = "off"``, so stale outputs are
    published verbatim and look authoritative -- the site once advertised a
    2.1.1 version banner and true-positive rates of 0.997, both artifacts of
    code that had since been fixed.

    The notebook is executed on a copy in memory. Nothing is written and no
    diff is produced: a mismatch simply fails, naming the cell.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    committed_nb = nbformat.read(notebook, as_version=4)
    fresh_nb = copy.deepcopy(committed_nb)
    nbclient.NotebookClient(
        fresh_nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook.parent)}},
    ).execute()

    stale = f"{notebook.name} is stale - re-run it and commit the outputs."
    for index, (committed, produced) in enumerate(
        zip(committed_nb.cells, fresh_nb.cells)
    ):
        if committed.cell_type != "code":
            continue
        want, got = _output_shape(committed), _output_shape(produced)
        if len(want) != len(got):
            pytest.fail(
                f"{stale}\ncell {index}: has {len(want)} committed output(s), "
                f"execution produced {len(got)}"
            )
        for (want_kind, want_text), (got_kind, got_text) in zip(want, got):
            if want_kind != got_kind:
                pytest.fail(
                    f"{stale}\ncell {index}: committed a {want_kind} output, "
                    f"execution produced {got_kind}"
                )
            if not _texts_agree(want_text, got_text):
                pytest.fail(
                    f"{stale}\ncell {index} output differs.\n"
                    f"--- committed ---\n{want_text}\n"
                    f"--- produced now ---\n{got_text}"
                )


# --------------------------------------------------------------------------
# Markdown code blocks
# --------------------------------------------------------------------------
#: Markdown files whose fenced ``python`` blocks form one running script.
#: Sphinx can single-source ``.rst`` snippets from ``examples/`` with
#: ``literalinclude``, but README.md is rendered by GitHub and PyPI, where that
#: directive does not exist -- so the only way to keep its code honest is to run
#: it. The blocks are executed in one shared namespace because later ones build
#: on earlier ones.
DOCS_WITH_CODE = [
    "README.md",
    "docs/index.rst",
    "docs/fairness_constraints.rst",
]

_PY_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
#: ``.. code-block:: python`` followed by an indented body. Options such as
#: ``:linenos:`` are skipped; the body is everything indented past the directive
#: until the first line that is neither blank nor indented.
_RST_BLOCK = re.compile(
    r"^([ \t]*)\.\. code-block:: python\n"  # the directive
    r"(?:\1[ \t]+:\S+:.*\n)*"  # any options
    r"\n"  # the blank line before the body
    r"((?:\1[ \t]+.*\n|[ \t]*\n)*)",  # the indented body
    re.M,
)


def _python_blocks(path: Path) -> list[str]:
    """Runnable python blocks, in document order, for markdown or reST."""
    text = path.read_text()
    if path.suffix == ".md":
        return _PY_BLOCK.findall(text)
    return [textwrap.dedent(body).strip("\n") for _, body in _RST_BLOCK.findall(text)]


@pytest.mark.parametrize("relative", DOCS_WITH_CODE)
def test_documentation_python_blocks_execute(relative: str) -> None:
    """Documentation code must run against the current API.

    This is the check that would have caught the README and quickstart snippets
    unpacking ``QSA`` into three values after it began returning ``QSAResult``:
    a break nothing executed, so nothing noticed.

    reST pages are covered as well as markdown. ``literalinclude`` single-sources
    a snippet from ``examples/``, but a snippet written inline in a ``.rst`` page
    -- the landing page's opening example, for one -- is executed by nothing else.
    """
    path = Path(__file__).resolve().parent.parent / relative
    blocks = _python_blocks(path)
    assert blocks, f"no python blocks found in {relative} - has the format changed?"

    namespace: dict = {}
    for index, block in enumerate(blocks):
        try:
            exec(compile(block, f"{relative} block {index}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - the failure message is the point
            pytest.fail(
                f"{relative} block {index} failed: {type(exc).__name__}: {exc}\n"
                f"--- block ---\n{block}"
            )
