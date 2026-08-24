"""Customize the fairness constraint and the confidence bound.

The behaviour of QSA is controlled by :class:`SeldonianConfig`:

* ``delta``           - the constraint holds with probability >= 1 - delta
* ``inequality``      - concentration inequality used for the bound
* ``candidate_ratio`` - fraction of training data used to pick the candidate
* ``constraint``      - the fairness constraint in reverse Polish (postfix)
  notation
* ``optimizer`` / ``max_iter`` / ``penalty`` - candidate-selection knobs

Constraints are written over two kinds of base variable. The **cells**
``TP(g)``, ``FP(g)``, ``TN(g)``, ``FN(g)`` are fractions of the whole group --
``TP(g)`` is the joint probability P(prediction = 1, label = 1 | group = g) --
and the four sum to 1 within a group. The **rates** ``TPR(g)``, ``FPR(g)``,
``TNR(g)``, ``FNR(g)`` condition on the label as well, so ``TPR(g)`` is the
true-positive rate P(prediction = 1 | label = 1, group = g), averaged over only
that group's positive rows. See ``cells_vs_rates.py`` for why the difference
matters.

Four inequalities are available. ``HOEFFDING_INEQUALITY``,
``EMPIRICAL_BERNSTEIN`` and ``BETTING`` are distribution-free; ``T_TEST``
assumes approximate normality of the sample mean, which is what makes the
result *quasi*-Seldonian.

For the common definitions there is no postfix string to write by hand:
:mod:`fair_seldonian.constraints.fairness` ships builders -
:func:`demographic_parity`, :func:`equal_opportunity` and
:func:`equalized_odds` - each taking a tolerance ``epsilon``.

Run it with::

    uv run python examples/custom_constraint.py
"""

from __future__ import annotations

from fair_seldonian import demographic_parity, equal_opportunity, equalized_odds
from fair_seldonian.algorithms import QSA
from fair_seldonian.config import SeldonianConfig
from fair_seldonian.constraints.inequalities import Inequality
from fair_seldonian.data import data_split, get_data


def evaluate(name: str, config: SeldonianConfig) -> None:
    data = get_data(
        N=20000,
        features=5,
        t_ratio=0.5,
        tp0_ratio=0.45,
        tp1_ratio=0.55,
        random_seed=7,
    )
    _, _, _, X_tr, Y_tr, T_tr = data_split(
        frac=0.6, all_data=data, random_state=1, m_test=0.3
    )
    result = QSA(X_tr, Y_tr, T_tr, "opt", None, None, config)

    print(f"{name}:")
    print(f"  delta={config.delta}, inequality={config.inequality.name}")
    print(f"  constraint = {config.constraint!r}")
    if result.passed_safety:
        # The number to quote is the bound from the safety set - that is the
        # evidence the certificate rests on. Re-evaluating on another split
        # gives a different number carrying no such claim.
        bound = result.diagnostics.safety_upper_bound
        print(f"  -> certified; safety bound <= {bound:+.4f}\n")
    else:
        print(f"  -> No Solution Found ({result.diagnostics.failure_mode})\n")


def main() -> None:
    # The default constraint is a gap between the TP *cells*, capped at 25% of
    # group 1's cell. Note that is not equal opportunity, which compares
    # true-positive rates; see the builders below.
    evaluate("Default config", SeldonianConfig())

    # Stricter: higher confidence (delta=0.01), Student's t-test bound, a larger
    # candidate split, and an absolute 0.10 cap on the cell gap.
    strict = SeldonianConfig(
        delta=0.01,
        inequality=Inequality.T_TEST,
        candidate_ratio=0.5,
        constraint="TP(1) TP(0) - abs 0.1 -",
    )
    evaluate("Stricter absolute-gap config", strict)

    # The named builders, each at the smallest tolerance that certifies on this
    # data. The ordering is informative and follows from how many leaves each
    # constraint has, since every leaf spends its own slice of delta. Equal
    # opportunity needs just two, TPR(1) and TPR(0), and no division. Demographic
    # parity needs four, TP and FP for each group. Equalized odds is loosest of
    # all: it sums two absolute gaps, so four leaves and both endpoints of each.
    evaluate("Equal opportunity", SeldonianConfig(constraint=equal_opportunity(0.10)))
    evaluate("Demographic parity", SeldonianConfig(constraint=demographic_parity(0.20)))
    evaluate("Equalized odds", SeldonianConfig(constraint=equalized_odds(0.25)))


if __name__ == "__main__":
    main()
