"""``TP(g)`` and ``TPR(g)`` are different quantities. Constraining them differs too.

The constraint notation offers two kinds of base variable, and mistaking one for
the other is easy because the names look alike:

``TP(g)`` -- a **cell**
    :math:`P(\\hat{Y}=1, Y=1 \\mid T=g)`, a joint probability. A fraction of the
    *whole* group: every row of the group contributes, rows with the wrong label
    contributing zero. The four cells sum to 1 within a group.

``TPR(g)`` -- a **rate**
    :math:`P(\\hat{Y}=1 \\mid Y=1, T=g)`, the true-positive rate. A mean over only
    that group's positive rows, so it carries its own, smaller sample size.

They coincide only when a group's base rate is 1. Whenever the two groups have
different base rates, equalising the cells is *not* equalising the rates -- so a
constraint written over cells and described as "equal opportunity" is enforcing
something else.

Run it with::

    uv run python examples/cells_vs_rates.py
"""

from __future__ import annotations

import numpy as np

from fair_seldonian.algorithms import QSA
from fair_seldonian.config import SeldonianConfig
from fair_seldonian.constraints import contributions, equal_opportunity
from fair_seldonian.data import data_split, get_data
from fair_seldonian.models import predict, simple_logistic


def main() -> None:
    # Deliberately unequal base rates: 30% of group 0 are positive, 70% of
    # group 1. This is exactly where cells and rates come apart.
    data = get_data(
        N=20000, features=5, t_ratio=0.5, tp0_ratio=0.3, tp1_ratio=0.7, random_seed=7
    )
    X_te, Y_te, T_te, X_tr, Y_tr, T_tr = data_split(
        frac=1.0, all_data=data, random_state=1, m_test=0.3
    )
    theta, theta1 = simple_logistic(X_tr, Y_tr)
    pred = predict(theta, theta1, X_te).detach()

    print("Plain logistic regression, groups with base rates 0.3 and 0.7\n")
    print(f"{'variable':<10} {'value':>8} {'samples':>9}   averaged over")
    print("-" * 60)
    for token, over in [
        ("TP(0)", "every row of group 0"),
        ("TP(1)", "every row of group 1"),
        ("TPR(0)", "group 0 rows with Y = 1"),
        ("TPR(1)", "group 1 rows with Y = 1"),
    ]:
        x = contributions(token, Y_te, pred, T_te)
        print(f"{token:<10} {float(x.mean()):>8.4f} {x.numel():>9}   {over}")

    cell_gap = abs(
        float(contributions("TP(1)", Y_te, pred, T_te).mean())
        - float(contributions("TP(0)", Y_te, pred, T_te).mean())
    )
    rate_gap = abs(
        float(contributions("TPR(1)", Y_te, pred, T_te).mean())
        - float(contributions("TPR(0)", Y_te, pred, T_te).mean())
    )
    print(f"\n  gap in cells |TP(1) - TP(0)|   = {cell_gap:.4f}")
    print(f"  gap in rates |TPR(1) - TPR(0)| = {rate_gap:.4f}")
    print("\nThe same model, the same data - two different disparities.")

    # Now constrain each and see that they ask for different models.
    print("\n" + "-" * 60)
    print("Constraining the cell gap vs the rate gap, tolerance 0.05:\n")
    for label, constraint in [
        ("cells  |TP(1) - TP(0)| <= 0.05", "TP(1) TP(0) - abs 0.05 -"),
        ("rates  (equal_opportunity)     ", equal_opportunity(0.05)),
    ]:
        config = SeldonianConfig(constraint=constraint)
        result = QSA(X_tr, Y_tr, T_tr, "opt", None, None, config)
        if result.passed_safety:
            p = predict(result.theta, result.theta1, X_te).detach()
            c = abs(
                float(contributions("TP(1)", Y_te, p, T_te).mean())
                - float(contributions("TP(0)", Y_te, p, T_te).mean())
            )
            r = abs(
                float(contributions("TPR(1)", Y_te, p, T_te).mean())
                - float(contributions("TPR(0)", Y_te, p, T_te).mean())
            )
            accuracy = float(
                ((p.numpy() >= 0.5).astype(int) == np.asarray(Y_te)).mean()
            )
            print(
                f"  {label}  -> certified;  cell gap {c:.4f}, "
                f"rate gap {r:.4f}, accuracy {accuracy:.3f}"
            )
        else:
            print(f"  {label}  -> {result.diagnostics.failure_mode}")

    print(
        "\nThe collapsed accuracy on the first row is not a bug, it is the point."
        "\nWith base rates this far apart the only way to equalise the *cells* is "
        "to\nstop predicting positives for group 1 almost entirely -- and doing so "
        "leaves\nthe true-positive *rate* gap essentially untouched. A constraint "
        "over cells\nlabelled 'equal opportunity' would therefore have paid the "
        "whole cost of\nfairness and delivered none of it."
    )
    print(
        "\nWriting a rate as the primitive TPR(g) rather than the ratio "
        "TP(g)/(TP(g)+FN(g))\nalso keeps the constraint free of division, which "
        "is what lets the `affine`\nbound apply to it."
    )


if __name__ == "__main__":
    main()
