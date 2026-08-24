"""Fail when the documentation stops describing the code.

Prose cannot be type-checked, so a rename or an added enum member leaves the docs
quietly wrong and every check still green. That is not hypothetical here: the
rate-primitives refactor left four separate places claiming that
``equal_opportunity`` and ``equalized_odds`` divide by per-group base rates, and
the fairness table still described demographic parity as ``TP + FP``, for a full
release.

The structural fix is generation -- the tables in ``docs/`` are emitted by
``docs/_ext/fair_seldonian_docs.py`` from the live package, so they cannot
disagree with it. These tests close the gap generation leaves open: a member that
no page mentions *at all* produces no wrong text, just missing text, and nothing
would notice.

They deliberately do not run Sphinx. Importing the extension module directly
keeps them in the fast unit-test job, where they run on every push.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import textwrap
from pathlib import Path

import pytest

from fair_seldonian.constraints.affine import NotAffine, compile_bounds
from fair_seldonian.constraints.expression_tree import (
    BASE_MEASURES,
    construct_expr_tree_base,
)
from fair_seldonian.constraints.fairness import FAIRNESS_CONSTRAINTS
from fair_seldonian.constraints.inequalities import INEQUALITY_INFO, Inequality
from fair_seldonian.models.logistic_regression import SELDONIAN_TYPES

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
SRC = REPO_ROOT / "src"


def _load_extension():
    """Import the Sphinx extension without starting Sphinx."""
    path = DOCS / "_ext" / "fair_seldonian_docs.py"
    spec = importlib.util.spec_from_file_location("fair_seldonian_docs", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("fair_seldonian_docs", module)
    spec.loader.exec_module(module)
    return module


def _all_docs_text() -> str:
    return "\n".join(p.read_text() for p in [*DOCS.glob("*.rst"), *DOCS.glob("*.md")])


# -- Nothing the library offers may be undocumented --------------------------


@pytest.mark.parametrize("member", list(Inequality), ids=lambda m: m.name)
def test_every_inequality_is_described(member: Inequality) -> None:
    """A new enum member must be described where the table is generated from."""
    assert member in INEQUALITY_INFO, (
        f"{member.name} has no INEQUALITY_INFO entry, so the generated table in "
        "docs/inequalities.rst would omit it. Add one beside the enum."
    )
    info = INEQUALITY_INFO[member]
    assert info.assumption and info.summary, f"{member.name} has empty metadata"


@pytest.mark.parametrize("member", list(Inequality), ids=lambda m: m.name)
def test_every_inequality_appears_in_the_docs(member: Inequality) -> None:
    assert member.name in _all_docs_text(), (
        f"{member.name} is never mentioned in docs/. Generation fills in a table "
        "row, but a member with no prose section is still undocumented."
    )


@pytest.mark.parametrize("name", sorted(SELDONIAN_TYPES))
def test_every_variant_appears_in_the_docs(name: str) -> None:
    assert SELDONIAN_TYPES[name].summary, f"{name} has an empty summary"
    assert f"``{name}``" in _all_docs_text(), (
        f"seldonian_type {name!r} is in the registry but no docs page mentions it."
    )


@pytest.mark.parametrize("name", sorted(FAIRNESS_CONSTRAINTS))
def test_every_fairness_builder_appears_in_the_docs(name: str) -> None:
    assert name in _all_docs_text(), (
        f"{name} is exported as a fairness builder but no docs page mentions it."
    )


@pytest.mark.parametrize("measure", BASE_MEASURES)
def test_every_base_variable_is_documented(measure: str) -> None:
    module = _load_extension()
    assert measure in module.BaseVariableTable._MEANING, (
        f"base variable {measure!r} has no description, so the generated DSL "
        "table in docs/fairness_constraints.rst would raise KeyError."
    )


# -- The generators themselves must work -------------------------------------


def test_generated_tables_build_without_error() -> None:
    """Exercise each directive's data path against the current package.

    Cheaper than a Sphinx build and catches the common breakage: a renamed
    attribute the extension reaches for.
    """
    module = _load_extension()
    for cls in (
        module.ConstraintTable,
        module.InequalityTable,
        module.VariantTable,
        module.BaseVariableTable,
    ):
        rows = cls.rows(cls)
        assert rows, f"{cls.__name__} produced no rows"
        width = len(cls.headers)
        for row in rows:
            assert len(row) == width, f"{cls.__name__} row {row!r} != {width} columns"


def test_measured_half_widths_rank_as_the_prose_claims() -> None:
    """The inequalities page says betting < empirical Bernstein < Hoeffding.

    That ordering is the reason to offer three distribution-free options, so pin
    it against the numbers the page actually renders rather than trusting prose.
    """
    module = _load_extension()
    w = module._measured_half_widths()
    shown = {k.name: round(v, 5) for k, v in w.items()}
    assert (
        w[Inequality.BETTING]
        < w[Inequality.EMPIRICAL_BERNSTEIN]
        < w[Inequality.HOEFFDING_INEQUALITY]
    ), f"documented ordering no longer holds: {shown}"


# -- Claims that previously drifted ------------------------------------------


@pytest.mark.parametrize("name", sorted(FAIRNESS_CONSTRAINTS))
def test_shipped_builders_stay_inside_the_affine_fragment(name: str) -> None:
    """The docs tell readers every builder can use the tighter affine bound.

    They are written over rate primitives specifically to avoid division. If one
    regressed to a ratio the claim would silently become false -- which is the
    inverse of the bug that prompted this file, where the docs claimed division
    that the code had already removed.
    """
    expr = FAIRNESS_CONSTRAINTS[name](0.1)
    assert "/" not in expr.split(), f"{name} emits a division: {expr}"
    try:
        compile_bounds(construct_expr_tree_base(expr))
    except NotAffine as exc:  # pragma: no cover - the assertion is the point
        pytest.fail(f"{name} no longer compiles to affine forms: {exc}")


def test_vendored_layout_still_matches_the_theme() -> None:
    """``docs/_templates/layout.html`` copies one block out of the theme.

    It re-declares pydata-sphinx-theme's ``extrahead`` so the three search
    scripts can be deferred; the theme injects them into every page with no
    loading attribute, which put ~100 kB of render-blocking JavaScript in front
    of every page view. A theme upgrade that adds or renames a script there
    would leave our copy loading the wrong set, silently.
    """
    pytest.importorskip("pydata_sphinx_theme", reason="docs extra not installed")
    import pydata_sphinx_theme

    theme = (
        Path(pydata_sphinx_theme.__file__).parent
        / "theme"
        / "pydata_sphinx_theme"
        / "layout.html"
    )
    block = re.search(
        r"{%-?\s*block extrahead\s*%}(.*?){%-?\s*endblock extrahead\s*%}",
        theme.read_text(),
        re.S,
    )
    assert block, "theme layout.html no longer defines an 'extrahead' block"
    upstream = set(re.findall(r"pathto\(\s*'([^']+)'", block.group(1)))
    ours = set(
        re.findall(
            r"pathto\(\s*'([^']+)'",
            (DOCS / "_templates" / "layout.html").read_text(),
        )
    )
    assert upstream == ours, (
        "pydata-sphinx-theme's extrahead block changed which scripts it loads.\n"
        f"  theme: {sorted(upstream)}\n"
        f"  ours:  {sorted(ours)}\n"
        "Re-sync docs/_templates/layout.html with the theme, keeping `defer`."
    )


def _leaf_modules() -> list[str]:
    """Importable, non-private modules that are not package ``__init__`` files."""
    import pkgutil

    import fair_seldonian

    out = []
    for mod in pkgutil.walk_packages(
        [str(Path(fair_seldonian.__file__).parent)], prefix="fair_seldonian."
    ):
        if any(part.startswith("_") for part in mod.name.split(".")) or mod.ispkg:
            continue
        out.append(mod.name)
    return sorted(out)


@pytest.mark.parametrize("modname", _leaf_modules())
def test_module_all_is_well_formed(modname: str) -> None:
    """``__all__`` decides what the API reference shows, so it must be honest."""
    import importlib

    mod = importlib.import_module(modname)
    names = getattr(mod, "__all__", None)
    assert names is not None, (
        f"{modname} has no __all__, so autodoc documents every public-looking "
        "name in it, including internal helpers."
    )
    for n in names:
        assert not n.startswith("_"), f"{modname}.__all__ lists a private name: {n}"
        assert hasattr(mod, n), f"{modname}.__all__ names {n!r}, which does not exist"


def test_every_exported_name_is_in_some_module_all() -> None:
    """A name re-exported from a package must be documented somewhere.

    ``__all__`` is what autodoc renders. Exporting a name from an ``__init__``
    while leaving it out of its module's ``__all__`` puts it in the public API
    and nowhere in the API reference -- invisible, but supported.
    """
    import importlib
    import inspect

    import fair_seldonian

    exported = {
        n: getattr(fair_seldonian, n)
        for n in dir(fair_seldonian)
        if not n.startswith("_")
    }
    for sub in ("algorithms", "config", "constraints", "data", "experiments", "models"):
        m = importlib.import_module(f"fair_seldonian.{sub}")
        exported.update({n: getattr(m, n) for n in dir(m) if not n.startswith("_")})

    missing = []
    for name, obj in sorted(exported.items()):
        home = getattr(obj, "__module__", None)
        if not home or not home.startswith("fair_seldonian") or inspect.ismodule(obj):
            continue
        names = getattr(importlib.import_module(home), "__all__", ())
        if name not in names:
            missing.append(f"{name} (exported, but not in {home}.__all__)")
    assert not missing, "public names absent from the API reference:\n  " + "\n  ".join(
        missing
    )


#: ``:math:`...``` and ``.. math::`` blocks, which KaTeX must be able to render.
_INLINE_MATH = re.compile(r":math:`([^`]+)`")
_BLOCK_MATH = re.compile(r"^\.\. math::\s*\n((?:\s*\n|[ \t]+.*\n)+)", re.M)


def _all_math() -> list[tuple[str, str, bool]]:
    """``(page, latex, is_display)`` for every equation in the prose docs."""
    out = []
    for p in sorted(DOCS.glob("*.rst")):
        text = p.read_text()
        out += [(p.name, m, False) for m in _INLINE_MATH.findall(text)]
        out += [
            (p.name, textwrap.dedent(m).strip(), True)
            for m in _BLOCK_MATH.findall(text)
        ]
    return out


def test_every_equation_renders_under_katex() -> None:
    """Math is pre-rendered by KaTeX at build time, and KaTeX is not MathJax.

    It covers a narrower slice of LaTeX, and ``sphinxcontrib-katex`` prerenders
    with ``throwOnError`` off -- so an expression it cannot parse is baked into
    the page as red error text and the build still succeeds. Render each one
    here with errors on, which is the only thing that actually fails.
    """
    katex = pytest.importorskip(
        "sphinxcontrib.katex", reason="docs extra not installed"
    )
    equations = _all_math()
    assert len(equations) > 100, (
        f"only found {len(equations)} equations - regex broken?"
    )

    broken = []
    for page, tex, display in equations:
        try:
            katex.render_latex(tex, {"throwOnError": True, "displayMode": display})
        except Exception as exc:  # noqa: BLE001 - any failure is a failure
            broken.append(f"{page}: {tex[:60]!r} -> {str(exc)[:90]}")
    assert not broken, "KaTeX cannot render:\n  " + "\n  ".join(broken)


#: The claim that was wrong in four places, in both wordings it appeared in:
#: "divide by per-group base rates" and "divide by a per-group base rate".
_DIVISION_CLAIM = re.compile(r"divid\w*\s+by\s+(?:a\s+)?per-group\s+base\s+rate")


def test_docs_do_not_claim_the_builders_divide() -> None:
    """Guard the wording, as a backstop to the structural check above.

    Matching a phrase is weaker than compiling the constraint, but it catches the
    case the structural test cannot: prose that is wrong about code that is right.
    """
    sources = [*DOCS.glob("*.rst"), *DOCS.glob("*.md"), *SRC.rglob("*.py")]
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in sources
        if _DIVISION_CLAIM.search(p.read_text())
    ]
    assert not offenders, (
        f"{offenders} say the builders divide by a per-group base rate. They use "
        "the TPR/FPR primitives, do not divide, and compile to affine forms."
    )
