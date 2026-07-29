"""Customize the fairness constraint and the confidence bound.

The behavior of QSA is controlled by :class:`SeldonianConfig`:

* ``delta``           - the constraint holds with probability >= 1 - delta
* ``inequality``      - concentration inequality used for the bound
* ``candidate_ratio`` - fraction of training data used to pick the candidate
* ``constraint``      - the fairness constraint in reverse Polish (postfix)
  notation, over group rates ``TP``, ``FP``, ``TN``, ``FN``

For the common definitions you don't need to write the postfix string by hand:
:mod:`fair_seldonian.constraints.fairness` ships ready-to-use builders -
:func:`demographic_parity`, :func:`equal_opportunity`, and
:func:`equalized_odds` - each taking a tolerance ``epsilon``. Below we compare a
hand-written string against these named helpers.

Run it with::

    uv run python examples/custom_constraint.py
"""

from __future__ import annotations

from fair_seldonian import demographic_parity, equal_opportunity, equalized_odds
from fair_seldonian.algorithms import QSA
from fair_seldonian.config import SeldonianConfig
from fair_seldonian.constraints.inequalities import Inequality
from fair_seldonian.data import data_split, get_data
from fair_seldonian.models import eval_ghat


def evaluate(name: str, config: SeldonianConfig) -> None:
    data = get_data(
        N=20000,
        features=5,
        t_ratio=0.5,
        tp0_ratio=0.45,
        tp1_ratio=0.55,
        random_seed=7,
    )
    X_te, Y_te, T_te, X_tr, Y_tr, T_tr = data_split(
        frac=0.6, all_data=data, random_state=1, m_test=0.3
    )
    theta, theta1, passed = QSA(X_tr, Y_tr, T_tr, "opt", None, None, config)

    print(f"{name}:")
    print(f"  delta={config.delta}, inequality={config.inequality.name}")
    print(f"  constraint = {config.constraint!r}")
    if passed:
        ub = float(eval_ghat(theta, theta1, X_te, Y_te, T_te, "opt", config))
        print(f"  -> certified; fairness upper bound <= {ub:.4f}\n")
    else:
        print("  -> No Solution Found\n")


def main() -> None:
    # Default configuration (relaxed equalized opportunity, Hoeffding, delta=0.05).
    evaluate("Default config", SeldonianConfig())

    # Stricter: higher confidence (delta=0.01), Student's t-test bound, a larger
    # candidate split, and an absolute 0.10 cap on the true-positive-rate gap.
    strict = SeldonianConfig(
        delta=0.01,
        inequality=Inequality.T_TEST,
        candidate_ratio=0.5,
        constraint="TP(1) TP(0) - abs 0.1 -",
    )
    evaluate("Stricter absolute-gap config", strict)

    # The same, using the named builders instead of a hand-written string. Each
    # takes a tolerance epsilon and defaults to comparing groups ("1", "0"). The
    # ratio-based criteria (equal opportunity, equalized odds) have looser
    # confidence bounds, so they need a larger epsilon to certify on this data.
    evaluate("Demographic parity", SeldonianConfig(constraint=demographic_parity(0.15)))
    evaluate("Equal opportunity", SeldonianConfig(constraint=equal_opportunity(0.2)))
    evaluate("Equalized odds", SeldonianConfig(constraint=equalized_odds(0.3)))


if __name__ == "__main__":
    main()
