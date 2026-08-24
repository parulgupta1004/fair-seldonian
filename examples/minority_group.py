"""What a small protected group costs, and which inequality softens it.

A confidence interval on a group's rate is built from the rows of *that group*.
Growing the dataset around a fixed-size minority therefore buys nothing for the
minority's interval -- and since the constraint's bound is only as tight as its
widest leaf, it buys little for the constraint either.

This is where the choice of concentration inequality earns its keep. Hoeffding
pays the a-priori range of the variable whatever the data does; empirical
Bernstein and betting pay for the variance they actually measure, which is small
when a group's rate sits far from 1/2 -- the common case for a minority.

Run it with::

    uv run python examples/minority_group.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fair_seldonian.constraints import check_constraint_groups
from fair_seldonian.constraints.inequalities import Inequality, eval_func_bound

INEQUALITIES = [
    Inequality.HOEFFDING_INEQUALITY,
    Inequality.EMPIRICAL_BERNSTEIN,
    Inequality.BETTING,
]


def build(n_total: int, n_minority: int, rate: float, seed: int = 0):
    """A dataset whose minority group has the given true-positive cell rate."""
    rng = np.random.default_rng(seed)
    T = pd.Series(np.array(["0"] * (n_total - n_minority) + ["1"] * n_minority))
    Y = np.concatenate(
        [
            rng.binomial(1, 0.5, n_total - n_minority),
            rng.binomial(1, rate, n_minority),
        ]
    )
    # A model that predicts the label exactly, so TP(1) is the minority's rate.
    pred = torch.tensor(Y.astype(float))
    return pd.Series(Y), pred, T


def main() -> None:
    print("Interval half-width for TP(1), the minority group's cell.\n")
    print("A fixed 200-member minority inside a growing dataset:\n")
    print(f"{'dataset':>9} {'minority':>9} {'half-width':>12}")
    print("-" * 34)
    for n_total in (2_000, 20_000, 200_000):
        Y, pred, T = build(n_total, 200, 0.5)
        lo, hi = eval_func_bound(
            "TP(1)",
            Y,
            pred,
            T,
            0.05,
            Inequality.HOEFFDING_INEQUALITY,
            None,
            False,
            False,
        )
        print(f"{n_total:>9} {200:>9} {(hi - lo) / 2:>12.4f}")
    print(
        "\nUnchanged: 100x the data, the same interval. Only the minority's own\n"
        "sample size moves it, which is why a fairness constraint naming a small\n"
        "group is expensive however large the dataset looks."
    )

    print("\n" + "-" * 60)
    print("Choice of inequality, 500-member minority, by the group's rate:\n")
    header = f"{'true rate':>10}" + "".join(
        f"{i.name.split('_')[0].lower():>16}" for i in INEQUALITIES
    )
    print(header)
    print("-" * len(header))
    for rate in (0.5, 0.2, 0.05):
        row = f"{rate:>10.2f}"
        for inequality in INEQUALITIES:
            Y, pred, T = build(20_000, 500, rate)
            lo, hi = eval_func_bound(
                "TP(1)", Y, pred, T, 0.05, inequality, None, False, False
            )
            row += f"{(hi - lo) / 2:>16.4f}"
        print(row)
    print(
        "\nHoeffding is flat: it assumes the worst case of 1/4 whatever the rate.\n"
        "The other two narrow sharply as the rate moves away from 1/2, which is\n"
        "exactly the regime a minority group tends to sit in.\n"
        "\nNote empirical Bernstein is the *widest* of the three at 0.50. It pays\n"
        "an additive penalty for estimating the variance, which only pays for\n"
        "itself once the measured variance is meaningfully below the worst case.\n"
        "Betting avoids that penalty and wins throughout."
    )

    # One more trap specific to group-based constraints.
    print("\n" + "-" * 60)
    print("Check your group labels before you trust a negative result:\n")
    try:
        check_constraint_groups(["0", "1"], np.array([0.0, 1.0, 0.0, 1.0]))
    except ValueError as exc:
        print(f"  {exc}")
    print(
        "\nLabels are matched as strings, so a T that has been upcast to float\n"
        "matches nothing. Every bound then fails closed to +inf and the run\n"
        "reports a clean 'No Solution Found' -- indistinguishable from a real one."
    )


if __name__ == "__main__":
    main()
