r"""Bound a constraint by compiling it to a max of affine forms.

The default path wraps a confidence interval around every node of the constraint
tree and combines them with interval arithmetic. That is sound but loose, for two
compounding reasons. Interval arithmetic assumes the worst about how the
sub-expressions relate, even when they are means over *disjoint* groups and hence
independent. And every leaf occurrence spends its own slice of ``delta``, so a
constraint mentioning three leaves pays a three-way union bound.

For constraints built from ``+``, ``-``, scaling by a constant and ``abs`` -
which covers the standard gap-based fairness definitions - neither cost is
necessary. Such an expression can be rewritten exactly as

.. math:: g(\theta) = \max_k \left( c_k + \sum_v a_{k,v} z_v \right)

a maximum of finitely many affine forms in the base variables. Each form can then
be bounded with a *single* interval, because all cells of one group are means over
the same rows: for group ``g`` the partial sum ``sum_v a_v z_v`` is itself the mean
over that group's samples of a scalar ``w_i = sum_v a_v x_i^{(v)}``. Distinct
groups are disjoint, so conditional on the group counts their means are
independent and Hoeffding applies to the weighted sum directly:

.. math:: \text{half-width} = \sqrt{\tfrac{1}{2}\ln(1/\delta')\sum_g r_g^2 / n_g}

with ``r_g`` the a-priori range of ``w_i`` within group ``g``. Only ``K`` slices of
``delta`` are spent, one per form, instead of one per leaf occurrence.

On the constraint ``|TP(0) - TP(1)| - 0.2 TP(1) <= 0`` this compiles to
``max(TP(0) - 1.2 TP(1), -TP(0) + 0.8 TP(1))`` - two forms rather than three leaf
intervals - and halves the slack over the true value. Measured at 50.0% across
sample sizes from 10k to 160k and several parameter vectors; the ratio barely
moves, because the saving comes from the delta split and the independence of the
groups rather than from the data. Since a Hoeffding half-width scales as
:math:`1/\sqrt{n}`, halving it is worth roughly 4x the data.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from scipy import stats

from .expression_tree import ExprTree, is_constant, is_func, is_mod, is_operator
from .inequalities import (
    Inequality,
    betting_interval,
    conditioning_set,
    contributions,
    parse_base_token,
)

if TYPE_CHECKING:
    from .._typing import Array, Bound


class AffineForm:
    """``constant + sum_v coefficient[v] * v`` over base-variable tokens."""

    __slots__ = ("constant", "coefficients")

    def __init__(
        self, constant: float = 0.0, coefficients: dict[str, float] | None = None
    ) -> None:
        self.constant = constant
        self.coefficients = dict(coefficients or {})

    def scaled(self, factor: float) -> AffineForm:
        return AffineForm(
            self.constant * factor,
            {v: c * factor for v, c in self.coefficients.items()},
        )

    def plus(self, other: AffineForm) -> AffineForm:
        merged = dict(self.coefficients)
        for v, c in other.coefficients.items():
            merged[v] = merged.get(v, 0.0) + c
        return AffineForm(self.constant + other.constant, merged)

    def __repr__(self) -> str:
        terms = " ".join(f"{c:+g}*{v}" for v, c in sorted(self.coefficients.items()))
        return f"AffineForm({self.constant:+g} {terms})"


class NotAffine(Exception):
    """The constraint uses an operation this compilation cannot represent."""


def compile_bounds(node: ExprTree | None) -> tuple[list[AffineForm], list[AffineForm]]:
    """Return ``(upper_forms, lower_forms)`` for the sub-expression at ``node``.

    ``max(upper_forms)`` is an exact upper envelope of the expression and
    ``min(lower_forms)`` a lower one. Both are returned because subtraction reads
    the lower envelope of its right operand.
    """
    if node is None:
        raise NotAffine("empty node")
    value = node.value

    if is_func(value):
        form = AffineForm(0.0, {value: 1.0})
        return [form], [form]
    if is_constant(value):
        form = AffineForm(float(value))
        return [form], [form]

    if is_mod(value):
        upper, lower = compile_bounds(node.left)
        # |x| <= max(U(x), -L(x)); and |x| >= 0, which is all we need downstream.
        return upper + [f.scaled(-1.0) for f in lower], [AffineForm(0.0)]

    if not is_operator(value) or node.left is None or node.right is None:
        raise NotAffine(f"unsupported node {value!r}")

    left_upper, left_lower = compile_bounds(node.left)
    right_upper, right_lower = compile_bounds(node.right)

    if value == "+":
        return (
            [a.plus(b) for a in left_upper for b in right_upper],
            [a.plus(b) for a in left_lower for b in right_lower],
        )
    if value == "-":
        return (
            [a.plus(b.scaled(-1.0)) for a in left_upper for b in right_lower],
            [a.plus(b.scaled(-1.0)) for a in left_lower for b in right_upper],
        )
    if value == "*":
        # Only scaling by a literal keeps the expression affine.
        if is_constant(node.right.value):
            factor = float(node.right.value)
            src_u, src_l = (
                (left_upper, left_lower) if factor >= 0 else (left_lower, left_upper)
            )
            return [f.scaled(factor) for f in src_u], [f.scaled(factor) for f in src_l]
        if is_constant(node.left.value):
            factor = float(node.left.value)
            src_u, src_l = (
                (right_upper, right_lower)
                if factor >= 0
                else (right_lower, right_upper)
            )
            return [f.scaled(factor) for f in src_u], [f.scaled(factor) for f in src_l]
        raise NotAffine("product of two non-constant sub-expressions")
    raise NotAffine(f"operator {value!r} is not affine")


#: ``PR`` and ``NR`` are sums of cells over the same rows, so they are rewritten
#: onto the cell basis before anything else. Every conditioning set is then
#: spanned by a genuine partition of the group's unit of mass, which is what
#: lets the range of ``w_i`` be read off the coefficients.
_EXPANSIONS = {"PR": ("TP", "FP"), "NR": ("TN", "FN")}

#: Within one conditioning set the basis functions partition each sample's unit
#: of mass, which is what lets the range of ``w_i`` be read off the coefficients.
_PARTITIONS = {
    None: ("TP", "FP", "TN", "FN"),
    1: ("TPR", "FNR"),
    0: ("FPR", "TNR"),
}


def _overlapping(keys: list[tuple[str, int | None]]) -> bool:
    """Whether any two conditioning sets share rows.

    Independence across the terms of a form is what licenses the weighted-sum
    bound, and it holds exactly when the conditioning sets are disjoint. Two sets
    on different groups never meet; on the same group, a whole-group set overlaps
    both label-conditioned ones, and identical sets are handled by combining
    coefficients before this check.
    """
    for i, (group_a, label_a) in enumerate(keys):
        for group_b, label_b in keys[i + 1 :]:
            if group_a != group_b:
                continue
            if label_a is None or label_b is None or label_a == label_b:
                return True
    return False


def form_upper_bound(
    form: AffineForm,
    Y: Array,
    predicted_Y: torch.Tensor,
    T: Array,
    delta: float,
    sample_scale: float = 1.0,
    inflate: float = 1.0,
    inequality: Inequality = Inequality.HOEFFDING_INEQUALITY,
) -> Bound:
    """One-sided upper bound on a single affine form.

    Hoeffding uses the a-priori range of ``w_i``; the ``t``-test and empirical
    Bernstein use its measured variance instead, which is much tighter when the
    per-sample values do not fill their range. Because the groups are independent
    given the counts, the variance of the whole form is just the sum of the
    per-group variances of the mean, so the same decomposition works for all
    three.

    ``sample_scale`` and ``inflate`` support the candidate-selection *prediction*
    of the safety-test bound: scale the group counts to the safety split's size and
    widen by the usual factor of 2. As elsewhere, that prediction is a heuristic and
    carries no guarantee.
    """
    groups: dict[tuple[str, int | None], dict[str, float]] = {}
    for token, coefficient in form.coefficients.items():
        measure, _ = parse_base_token(token)
        bucket = groups.setdefault(conditioning_set(token), {})
        # Accumulate rather than assign: after expansion two distinct tokens can
        # land on the same cell, as PR(g) and TP(g) both do on TP.
        for cell in _EXPANSIONS.get(measure, (measure,)):
            bucket[cell] = bucket.get(cell, 0.0) + coefficient
    if _overlapping(list(groups)):
        raise NotAffine(
            "form mixes base variables whose conditioning sets overlap "
            f"({sorted(form.coefficients)}); their means are not independent, so "
            "the weighted-sum bound does not apply"
        )
    n_forms_hint = max(len(groups), 1)

    total = form.constant
    variance_term = 0.0
    total_n = 0.0
    n_groups = 0
    betting_width = 0.0
    for (g, label), cell_coefficients in groups.items():
        partition = _PARTITIONS[label]
        first = next(iter(cell_coefficients))
        n_g = int(contributions(f"{first}({g})", Y, predicted_Y, T).numel())
        if n_g == 0:
            return math.inf
        # w_i = sum_v coeff * x_i(v). The basis functions of a conditioning set
        # partition each sample's unit of mass, so w_i lies between the smallest
        # and largest coefficient (absent ones contribute 0).
        per_sample = torch.zeros(n_g, dtype=torch.float64)
        for cell in partition:
            coefficient = cell_coefficients.get(cell)
            if coefficient:
                token = f"{cell}({g})"
                per_sample = per_sample + coefficient * contributions(
                    token, Y, predicted_Y, T
                ).to(torch.float64)
        # The bound is a number, not a graph node; see eval_ghat.
        per_sample = per_sample.detach()
        if inequality == Inequality.BETTING:
            # Betting works on values in [0, 1], and does not decompose across
            # groups the way a variance does, so each group's contribution is
            # bounded separately - rescaled into the unit interval and back - and
            # the half-widths are summed. That forgoes the independence saving,
            # but betting's adaptivity to the observed spread usually more than
            # repays it.
            present = [cell_coefficients.get(cell, 0.0) for cell in partition]
            span = max(max(present), 0.0) - min(min(present), 0.0)
            offset = min(min(present), 0.0)
            values = per_sample.detach().numpy()
            if span <= 0:
                total += float(per_sample.mean())
                continue
            unit = (values - offset) / span
            lo, hi = betting_interval(unit, delta / n_forms_hint, two_sided=False)
            if not math.isfinite(hi):
                return math.inf
            total += float(per_sample.mean())
            betting_width += span * (hi - float(unit.mean()))
            continue
        total += float(per_sample.mean())
        effective_n = n_g * sample_scale
        if inequality == Inequality.HOEFFDING_INEQUALITY:
            present = [cell_coefficients.get(cell, 0.0) for cell in partition]
            width = max(max(present), 0.0) - min(min(present), 0.0)
            variance_term += width * width / effective_n
        else:
            variance_term += float(per_sample.var(unbiased=True)) / effective_n
        total_n += effective_n
        n_groups += 1

    if inequality == Inequality.BETTING:
        return total + inflate * betting_width
    if inequality == Inequality.HOEFFDING_INEQUALITY:
        half_width = math.sqrt(0.5 * math.log(1.0 / delta) * variance_term)
    elif inequality == Inequality.T_TEST:
        degrees = max(int(total_n) - n_groups, 1)
        half_width = float(stats.t.ppf(1.0 - delta, degrees)) * math.sqrt(variance_term)
    elif inequality == Inequality.EMPIRICAL_BERNSTEIN:
        log_term = math.log(1.0 / delta)
        half_width = math.sqrt(2 * variance_term * log_term) + (
            7 * log_term / (3 * max(total_n - 1, 1))
        )
    else:
        raise ValueError(f"Unknown inequality: {inequality!r}")
    return total + inflate * half_width


def affine_upper_bound(
    root: ExprTree,
    Y: Array,
    predicted_Y: torch.Tensor,
    T: Array,
    delta: float,
    sample_scale: float = 1.0,
    inflate: float = 1.0,
    inequality: Inequality = Inequality.HOEFFDING_INEQUALITY,
) -> Bound:
    """Upper bound on the constraint via its max-of-affine-forms compilation.

    The budget is split evenly across the ``K`` forms. Even splitting is not
    optimal - the forms have different widths, so a convex allocation would do
    slightly better - but the dominant saving comes from having ``K`` intervals
    instead of one per leaf occurrence.

    :type root: ~fair_seldonian.constraints.expression_tree.ExprTree
    """
    upper_forms, _ = compile_bounds(root)
    if not upper_forms:
        raise NotAffine("no forms produced")
    per_form_delta = delta / len(upper_forms)
    return max(
        form_upper_bound(
            form, Y, predicted_Y, T, per_form_delta, sample_scale, inflate, inequality
        )
        for form in upper_forms
    )
