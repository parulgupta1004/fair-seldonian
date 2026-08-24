"""Guards on what the package exposes and what the docs claim it exposes.

Two failures this suite could not previously see:

* Nothing else in this repo imports from ``fair_seldonian`` directly -- the
  tests and examples reach into the submodules -- so dropping a name from an
  ``__init__`` breaks downstream code without breaking anything here.
* A module can be complete, tested and reachable only by its full dotted path.
  ``constraints.affine`` shipped that way -- absent from every ``__init__`` and
  from the Sphinx API reference, so nothing but the source tree revealed it.
"""

from __future__ import annotations

import ast
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
