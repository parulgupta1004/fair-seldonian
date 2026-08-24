"""Guards on what the package exposes and what the docs claim it exposes.

Three failures this suite could not previously see:

* A module can be complete, tested and reachable only by its full dotted path.
  ``constraints.affine`` shipped that way -- absent from every ``__init__`` and
  from the Sphinx API reference, so nothing but the source tree revealed it.
* A ``:func:`` cross-reference to a renamed function degrades to plain text
  rather than an error. ``conf.py`` sets ``suppress_warnings`` for missing
  MyST xrefs and nitpicky mode is off, so even ``sphinx-build -W`` stays green
  while the docs point at a name that no longer exists (``fHat`` did, for
  several releases).
* Dropping a name from an ``__init__`` breaks downstream imports without
  breaking anything in this repo, which imports from the submodules.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import fair_seldonian

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
PACKAGE_DIR = Path(fair_seldonian.__file__).parent

# Names the package promises at its top level. Removing one is a breaking change
# for downstream code, so it should require editing this list on purpose.
PUBLIC_API = frozenset(
    {
        # algorithm
        "QSA",
        "QSAResult",
        "Diagnostics",
        "safety_test",
        # configuration
        "SeldonianConfig",
        "DEFAULT_CONFIG",
        # fairness constraint builders
        "FAIRNESS_CONSTRAINTS",
        "demographic_parity",
        "equal_opportunity",
        "equalized_odds",
        "error_rate",
        "error_rate_parity",
        "validate_constraint",
        # constraint machinery
        "Inequality",
        "construct_expr_tree",
        "construct_expr_tree_base",
        "eval_expr_tree",
        "eval_expr_tree_base",
        "eval_expr_tree_conf_interval",
        "eval_expr_tree_conf_interval_base",
        "inorder",
        # bounds
        "NotAffine",
        "affine_upper_bound",
        "compile_bounds",
        "betting_interval",
        "contributions",
        "check_constraint_groups",
        "constraint_groups",
        # data
        "get_data",
        "data_split",
        # models
        "simple_logistic",
        "predict",
        "f_hat",
        "ghat",
        "eval_ghat",
        # experiment harness
        "StudySpec",
        "run_study",
        "summarise",
        "clopper_pearson",
    }
)


@pytest.mark.parametrize("name", sorted(PUBLIC_API))
def test_public_name_is_exported(name: str) -> None:
    assert hasattr(fair_seldonian, name), (
        f"fair_seldonian.{name} is documented as public but is not exported. "
        "Removing it is a breaking change; update PUBLIC_API deliberately."
    )


def test_no_undeclared_public_names() -> None:
    """The root namespace should not grow by accident.

    A stray ``import numpy`` in an ``__init__`` leaks ``fair_seldonian.numpy``
    into the public surface, which people then import.
    """
    actual = {n for n in dir(fair_seldonian) if not n.startswith("_")}
    submodules = {m.name for m in pkgutil.iter_modules([str(PACKAGE_DIR)])}
    unexpected = actual - PUBLIC_API - submodules
    assert not unexpected, (
        f"undeclared names in the root namespace: {sorted(unexpected)}"
    )


def _public_modules() -> list[str]:
    """Every importable, non-private module in the package."""
    found = []
    for mod in pkgutil.walk_packages([str(PACKAGE_DIR)], prefix="fair_seldonian."):
        if any(part.startswith("_") for part in mod.name.split(".")):
            continue
        found.append(mod.name)
    return sorted(found)


def test_every_public_module_is_in_the_api_reference() -> None:
    """Sphinx renders only what an ``automodule`` directive names.

    ``constraints.affine`` was missing, so the module implementing the tightest
    bound the library offers did not appear in the published API docs at all.
    """
    documented = set()
    for rst in (DOCS / "api").glob("*.rst"):
        documented.update(
            re.findall(r"^\.\.\s+automodule::\s+(\S+)", rst.read_text(), re.M)
        )
    missing = [m for m in _public_modules() if m not in documented]
    assert not missing, (
        f"public modules absent from docs/api/: {missing}. "
        "Add an automodule directive so they appear in the API reference."
    )


def _doc_cross_references() -> list[tuple[Path, str]]:
    """``:func:`fair_seldonian.x.y``` style targets across the prose docs."""
    pattern = re.compile(
        r":(?:mod|class|func|attr|meth|data|exc):`~?([A-Za-z_][\w.]*)`"
    )
    refs = []
    for path in sorted([*DOCS.glob("*.rst"), *DOCS.glob("*.md")]):
        for target in pattern.findall(path.read_text()):
            if target.startswith("fair_seldonian"):
                refs.append((path, target))
    return refs


def test_doc_cross_references_resolve() -> None:
    """Every ``fair_seldonian.*`` target the docs name must actually exist.

    Sphinx will not tell us: an unresolved Python xref renders as plain text,
    and the docs build passes even under ``-W``.
    """
    broken = []
    for path, target in _doc_cross_references():
        parts = target.split(".")
        obj = None
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except ModuleNotFoundError:
                continue
            rest = parts[i:]
            break
        else:
            broken.append(f"{path.name}: {target} (no such module)")
            continue
        for attr in rest:
            if not hasattr(obj, attr):
                broken.append(f"{path.name}: {target} (no attribute {attr!r})")
                break
            obj = getattr(obj, attr)
    assert not broken, "docs reference names that do not exist:\n  " + "\n  ".join(
        broken
    )


def test_doc_cross_reference_scan_finds_something() -> None:
    """Stop the check above from passing because the regex matched nothing."""
    assert len(_doc_cross_references()) >= 5


@pytest.mark.parametrize(
    "init", sorted(PACKAGE_DIR.rglob("__init__.py")), ids=lambda p: p.name
)
def test_reexports_use_explicit_alias_form(init: Path) -> None:
    """``from .x import y as y`` marks a deliberate re-export.

    Without the redundant alias, type checkers treat the name as a private
    import and downstream ``from fair_seldonian import y`` fails type checking
    even though it works at runtime.
    """
    tree = ast.parse(init.read_text())
    bare = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname is None
    ]
    assert not bare, f"{init.relative_to(PACKAGE_DIR)} re-exports without `as`: {bare}"
