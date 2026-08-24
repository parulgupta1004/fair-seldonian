"""Sphinx directives that generate documentation tables from the package itself.

Every hand-maintained table in these docs had drifted from the code by the time
this was written: ``fairness_constraints.md`` still described demographic parity
as ``TP + FP`` and equal opportunity as ``TP / (TP + FN)``, encodings replaced by
the ``PR``/``TPR`` primitives, and three separate pages claimed that equal
opportunity and equalized odds fall outside the affine fragment when in fact all
four builders compile to affine forms.

None of that is caught by a docs build, because prose cannot be type-checked. The
fix is to stop writing the facts down: each directive here calls the library and
renders whatever it actually reports, so a refactor updates the docs by
construction. The remaining risk -- a new enum member or builder that no page
mentions at all -- is covered by ``tests/test_docs_sync.py``.

Directives raise rather than degrade if introspection fails, so a build under
``-W`` fails loudly instead of silently emitting an empty table.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

from fair_seldonian.constraints.affine import NotAffine, compile_bounds
from fair_seldonian.constraints.expression_tree import (
    BASE_MEASURES,
    construct_expr_tree_base,
    is_func,
)
from fair_seldonian.constraints.fairness import FAIRNESS_CONSTRAINTS, error_rate
from fair_seldonian.constraints.inequalities import (
    INEQUALITY_INFO,
    Inequality,
    betting_interval,
    conditioning_set,
    eval_empirical_bernstein,
    eval_hoeffding,
    eval_t_test,
)
from fair_seldonian.models.logistic_regression import SELDONIAN_TYPES

#: Tolerance used when rendering an example constraint string. Arbitrary, but the
#: same everywhere so the tables agree with each other.
EXAMPLE_EPSILON = 0.1

#: Fixed setting for the worked half-widths in the inequality table. A base rate
#: of 0.1 is the interesting case: it is where paying for measured variance rather
#: than the worst case of 1/4 actually buys something.
DEMO_N = 2000
DEMO_DELTA = 0.05
DEMO_RATE = 0.1


def _first_paragraph(obj: Any) -> str:
    """The docstring's opening paragraph, whitespace-collapsed onto one line."""
    doc = inspect.getdoc(obj) or ""
    para: list[str] = []
    for line in doc.splitlines():
        if not line.strip():
            if para:
                break
            continue
        para.append(line.strip())
    return " ".join(para)


def _references(obj: Any) -> list[str]:
    """Entries under a ``References:`` block in the docstring."""
    doc = inspect.getdoc(obj) or ""
    out: list[str] = []
    inside = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith("References:"):
            inside = True
            continue
        if inside:
            if stripped.startswith("- "):
                out.append(stripped[2:])
            elif stripped.startswith(":") or (stripped and not line.startswith(" ")):
                break
            elif stripped and out:
                out[-1] += " " + stripped
    return out


def _escape(text: str) -> str:
    """Protect a generated cell from being reparsed as RST markup."""
    return text.replace("\\", "\\\\").replace("*", r"\*").replace("|", r"\|")


class _GeneratedTable(Directive):
    """Base for directives that emit a ``list-table`` built from live code."""

    has_content = False
    headers: tuple[str, ...] = ()
    widths: tuple[int, ...] = ()

    def rows(self) -> list[tuple[str, ...]]:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self) -> list[nodes.Node]:
        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :class: fs-generated",
        ]
        if self.widths:
            lines.append("   :widths: " + " ".join(str(w) for w in self.widths))
        lines.append("")
        for row in (self.headers, *self.rows()):
            for i, cell in enumerate(row):
                lines.append(f"   {'*' if i == 0 else ' '} - {cell}")
        container = nodes.container()
        self.state.nested_parse(
            StringList(lines, source="<fair_seldonian_docs>"), 0, container
        )
        return container.children


class ConstraintTable(_GeneratedTable):
    """Every shipped fairness builder, with the string it actually produces."""

    headers = ("Builder", "Postfix at ``epsilon=0.1``", "Leaves", "Affine forms")
    widths = (22, 44, 10, 14)

    def rows(self) -> list[tuple[str, ...]]:
        out = []
        for name, builder in _all_builders():
            expr = builder(EXAMPLE_EPSILON)
            leaves = sum(1 for token in expr.split() if is_func(token))
            try:
                forms, _ = compile_bounds(construct_expr_tree_base(expr))
                affine = str(len(forms))
            except NotAffine:
                affine = "not affine"
            out.append(
                (
                    f":func:`~fair_seldonian.constraints.fairness.{name}`",
                    f"``{expr}``",
                    str(leaves),
                    affine,
                )
            )
        return out


class ConstraintCards(Directive):
    """One card per fairness builder: what it bounds, and its references."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        lines = [".. grid:: 1 1 2 2", "   :gutter: 3", ""]
        for name, builder in _all_builders():
            expr = builder(EXAMPLE_EPSILON)
            try:
                forms, _ = compile_bounds(construct_expr_tree_base(expr))
                affine = f"affine, {len(forms)} form(s)"
            except NotAffine:
                affine = "outside the affine fragment"
            lines += [
                f"   .. grid-item-card:: {name}",
                "      :class-card: sd-shadow-sm",
                "",
                f"      {_first_paragraph(builder)}",
                "",
                f"      ``{expr}``",
                "",
                f"      *{affine}*",
                "",
            ]
            for ref in _references(builder):
                lines.append(f"      - {ref}")
            lines.append("")
        container = nodes.container()
        self.state.nested_parse(
            StringList(lines, source="<fair_seldonian_docs>"), 0, container
        )
        return container.children


class InequalityTable(_GeneratedTable):
    """The concentration inequalities, with a worked half-width for each."""

    headers = (
        "Inequality",
        "Assumption",
        "Guarantee",
        f"Half-width at n={DEMO_N}, rate {DEMO_RATE}",
    )
    widths = (24, 24, 18, 22)

    def rows(self) -> list[tuple[str, ...]]:
        widths = _measured_half_widths()
        out = []
        for member in Inequality:
            info = INEQUALITY_INFO[member]
            guarantee = (
                "distribution-free" if info.distribution_free else "asymptotic only"
            )
            out.append(
                (
                    f"``{member.name}``",
                    _escape(info.assumption),
                    guarantee,
                    f"{widths[member]:.4f}",
                )
            )
        return out


class VariantTable(_GeneratedTable):
    """Every accepted ``seldonian_type``, from the dispatch registry."""

    headers = ("Mode", "What it changes")
    widths = (14, 86)

    def rows(self) -> list[tuple[str, ...]]:
        return [
            (f"``{name}``", _escape(spec.summary))
            for name, spec in SELDONIAN_TYPES.items()
        ]


class BaseVariableTable(_GeneratedTable):
    """The constraint DSL's base variables and the rows each averages over."""

    headers = ("Primitive", "Conditions on", "Averaged over")
    widths = (18, 26, 56)

    _KIND = {
        None: "group only",
        1: "group and ``Y = 1``",
        0: "group and ``Y = 0``",
    }
    _MEANING = {
        "TP": "predicted 1 and labelled 1, as a fraction of the whole group",
        "FP": "predicted 1 and labelled 0, as a fraction of the whole group",
        "TN": "predicted 0 and labelled 0, as a fraction of the whole group",
        "FN": "predicted 0 and labelled 1, as a fraction of the whole group",
        "TPR": "recall: predicted 1 among the group's positives",
        "FPR": "predicted 1 among the group's negatives",
        "TNR": "predicted 0 among the group's negatives",
        "FNR": "predicted 0 among the group's positives",
        "PR": "predicted 1, whatever the label; equals ``TP + FP``",
        "NR": "predicted 0, whatever the label; equals ``TN + FN``",
    }

    def rows(self) -> list[tuple[str, ...]]:
        out = []
        for measure in BASE_MEASURES:
            _, label = conditioning_set(f"{measure}(g)")
            out.append(
                (
                    f"``{measure}(g)``",
                    self._KIND[label],
                    self._MEANING[measure],
                )
            )
        return out


def _all_builders() -> list[tuple[str, Any]]:
    """The parity builders plus ``error_rate``, which has a different signature."""
    return [*FAIRNESS_CONSTRAINTS.items(), ("error_rate", error_rate)]


def _measured_half_widths() -> dict[Inequality, float]:
    """Evaluate each inequality on one fixed sample so 'tighter' is measured.

    Uses a deterministic Bernoulli draw rather than quoted figures, so the column
    cannot disagree with the implementations it describes.
    """
    rng = np.random.default_rng(0)
    x = rng.binomial(1, DEMO_RATE, DEMO_N).astype(float)
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    lo, hi = betting_interval(x, DEMO_DELTA, True)
    return {
        Inequality.HOEFFDING_INEQUALITY: eval_hoeffding(mean, DEMO_N, DEMO_DELTA, True)[
            1
        ]
        - mean,
        Inequality.EMPIRICAL_BERNSTEIN: eval_empirical_bernstein(
            mean, var, DEMO_N, DEMO_DELTA, True
        )[1]
        - mean,
        Inequality.T_TEST: eval_t_test(mean, var**0.5, DEMO_N, DEMO_DELTA, True)[1]
        - mean,
        Inequality.BETTING: hi - mean,
    }


def setup(app: Any) -> dict[str, Any]:
    app.add_directive("fs-constraint-table", ConstraintTable)
    app.add_directive("fs-constraint-cards", ConstraintCards)
    app.add_directive("fs-inequality-table", InequalityTable)
    app.add_directive("fs-variant-table", VariantTable)
    app.add_directive("fs-basevar-table", BaseVariableTable)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
