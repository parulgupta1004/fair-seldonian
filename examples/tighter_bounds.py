"""Compare how tightly each bound certifies the same model.

Everything the Seldonian guarantee costs shows up as *slack*: the gap between
the upper bound QSA can certify and the constraint's true value. Slack is what
decides how much data you need before a fair model becomes certifiable at all,
so it is worth knowing where it comes from.

Two knobs move it independently. ``seldonian_type`` chooses how the bound is
assembled from the constraint tree, and ``inequality`` chooses the concentration
bound used at each leaf. This script holds the model fixed -- so the optimizer
plays no part -- and reports the slack for each combination.

Run it with::

    uv run python examples/tighter_bounds.py
"""

from __future__ import annotations

from fair_seldonian.config import SeldonianConfig
from fair_seldonian.constraints import construct_expr_tree_base, eval_expr_tree_base
from fair_seldonian.constraints.inequalities import Inequality
from fair_seldonian.data import data_split, get_data
from fair_seldonian.models import eval_ghat, predict, simple_logistic

CONSTRAINT = "TP(1) TP(0) - abs 0.25 TP(1) * -"

MODES = [
    ("base", "one interval per tree node, uniform delta"),
    ("opt", "constant-aware delta + union-bound merge"),
    ("affine", "max of affine forms, one interval per form"),
]
INEQUALITIES = [
    Inequality.HOEFFDING_INEQUALITY,
    Inequality.EMPIRICAL_BERNSTEIN,
    Inequality.BETTING,
]


def main() -> None:
    data = get_data(
        N=20000, features=5, t_ratio=0.5, tp0_ratio=0.45, tp1_ratio=0.55, random_seed=7
    )
    X_te, Y_te, T_te, X_tr, Y_tr, T_tr = data_split(
        frac=1.0, all_data=data, random_state=1, m_test=0.3
    )

    # A fixed model, so every number below differs only in how the bound is
    # built -- not in which model was found.
    theta, theta1 = simple_logistic(X_tr, Y_tr)

    truth = float(
        eval_expr_tree_base(
            construct_expr_tree_base(CONSTRAINT),
            Y_te,
            # detach: simple_logistic returns parameters that carry gradients,
            # and the plug-in value is a plain number, not part of a graph.
            predict(theta, theta1, X_te).detach(),
            T_te,
        )
    )
    print(f"true g(theta) on held-out data: {truth:+.4f}")
    print("(a bound certifies the constraint when it is <= 0; slack = bound - truth)\n")

    header = f"{'bound':<10}" + "".join(
        f"{i.name.split('_')[0]:>16}" for i in INEQUALITIES
    )
    print(header)
    print("-" * len(header))

    baseline: dict[Inequality, float] = {}
    for mode, _ in MODES:
        row = f"{mode:<10}"
        for inequality in INEQUALITIES:
            config = SeldonianConfig(constraint=CONSTRAINT, inequality=inequality)
            bound = float(eval_ghat(theta, theta1, X_te, Y_te, T_te, mode, config))
            slack = bound - truth
            baseline.setdefault(inequality, slack)
            row += f"{slack:>10.4f} ({100 * slack / baseline[inequality]:>3.0f}%)"
        print(row)

    print("\nEach cell is the slack; the percentage is relative to the `base` row")
    print("for that inequality, so lower is tighter.\n")
    for mode, note in MODES:
        print(f"  {mode:<8} {note}")
    print(
        "\nHalving the slack is worth roughly four times the data, since a "
        "Hoeffding\nhalf-width shrinks as 1/sqrt(n)."
    )

    # The affine path only accepts constraints built from +, -, scaling by a
    # constant and abs. Anything else is refused rather than silently widened.
    from fair_seldonian.constraints.affine import NotAffine

    ratio_constraint = "TP(1) TP(1) FN(1) + / 0.1 -"
    try:
        eval_ghat(
            theta,
            theta1,
            X_te,
            Y_te,
            T_te,
            "affine",
            SeldonianConfig(constraint=ratio_constraint),
        )
    except NotAffine as exc:
        print(f"\n`affine` refuses {ratio_constraint!r}:\n  {exc}")
        print("  Use the TPR(g) primitive instead of the ratio to stay affine.")


if __name__ == "__main__":
    main()
