"""Adult experiment: demographic parity under a Seldonian guarantee.

Two caveats are inherent to the dataset and are reported rather than hidden.
Adult is a fixed finite sample, so trials resample from one pool and are *not*
independent draws the way the synthetic trials are; the error bars understate
sampling variability accordingly. And Adult should be considered retired (Ding et
al., 2021) - it is used here because it remains the common point of comparison
and because its group rates exercise a regime the synthetic data does not.
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fair_seldonian.algorithms.qsa import QSA  # noqa: E402
from fair_seldonian.config import SeldonianConfig  # noqa: E402
from fair_seldonian.constraints.expression_tree import (  # noqa: E402
    construct_expr_tree_base,
    eval_expr_tree_base,
)
from fair_seldonian.constraints.inequalities import (  # noqa: E402
    Inequality,
    check_constraint_groups,
)
from fair_seldonian.experiments.plots import plot_all  # noqa: E402
from fair_seldonian.experiments.results import save_summary, summarise  # noqa: E402
from fair_seldonian.models.logistic_regression import (  # noqa: E402
    f_hat,
    predict,
    simple_logistic,
)

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


#: |P(Yhat=1 | female) - P(Yhat=1 | male)| - epsilon <= 0, written over cells.
#: Predicted-positive rate within a group is TP(g) + FP(g).
def demographic_parity(epsilon: float, g1: str = "F", g0: str = "M") -> str:
    return f"TP({g1}) FP({g1}) + TP({g0}) FP({g0}) + - abs {epsilon!r} -"


def load_adult(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(
        path, header=None, names=COLUMNS, skipinitialspace=True, na_values="?"
    ).dropna()
    Y = (df["income"].str.strip().str.rstrip(".") == ">50K").astype(int).to_numpy()
    T = np.where(df["sex"].str.strip() == "Female", "F", "M")

    numeric = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    # education_num already encodes education ordinally, and native_country is
    # overwhelmingly one value spread over 40 sparse dummies, so both are dropped.
    # Occupation is kept: it carries real income signal.
    categorical = [
        "workclass",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
    ]
    X_num = df[numeric].to_numpy(dtype=float)
    X_num = (X_num - X_num.mean(0)) / X_num.std(0)
    X_cat = pd.get_dummies(df[categorical], drop_first=True).to_numpy(dtype=float)
    X = np.hstack([X_num, X_cat])
    return X, Y, T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--tag", default="adult")
    parser.add_argument(
        "--variants", nargs="*", default=None, help="subset of base/ebernstein/affine"
    )
    args = parser.parse_args()

    X, Y, T = load_adult(args.data)
    constraint = demographic_parity(args.epsilon)
    check_constraint_groups(["F", "M"], T)
    print(
        f"Adult: {len(Y)} rows, {X.shape[1]} features, "
        f"P(Y=1|F)={Y[T == 'F'].mean():.4f} P(Y=1|M)={Y[T == 'M'].mean():.4f}"
    )

    rng = np.random.default_rng(0)
    holdout = rng.permutation(len(Y))[: len(Y) // 5]
    pool = np.setdiff1d(np.arange(len(Y)), holdout)
    eval_X, eval_Y, eval_T = X[holdout], Y[holdout], T[holdout]

    tree = construct_expr_tree_base(constraint)
    ms = [2_000, 5_000, 10_000, 15_000, 20_000, len(pool)]

    all_variants = [
        ("base", "base", Inequality.HOEFFDING_INEQUALITY),
        ("ebernstein", "base", Inequality.EMPIRICAL_BERNSTEIN),
        ("affine", "affine", Inequality.HOEFFDING_INEQUALITY),
    ]
    selected = [
        v for v in all_variants if args.variants is None or v[0] in args.variants
    ]
    for key, variant, inequality in selected:
        name = f"{args.tag}/{key}"
        config = SeldonianConfig(
            constraint=constraint,
            delta=0.05,
            candidate_ratio=0.60,
            inequality=inequality,
        )
        recs = {
            k: np.full((args.trials, len(ms)), np.nan)
            for k in (
                "solution_found",
                "log_loss",
                "g_value",
                "failure_mode",
                "ls_log_loss",
                "ls_g_value",
            )
        }
        recs["solution_found"][:] = 0.0
        recs["failure_mode"][:] = 0.0

        for trial in range(args.trials):
            trial_rng = np.random.default_rng(1000 + trial)
            for i, m in enumerate(ms):
                # Adult is not shuffled; an unshuffled candidate/safety split would
                # not be exchangeable, which breaks the i.i.d. premise.
                idx = trial_rng.permutation(pool)[:m]
                Xs, Ys, Ts = X[idx], Y[idx], T[idx]

                ls_theta, ls_theta1 = simple_logistic(Xs, Ys)
                recs["ls_log_loss"][trial, i] = -float(
                    f_hat(ls_theta, ls_theta1, eval_X, eval_Y)
                )
                recs["ls_g_value"][trial, i] = float(
                    eval_expr_tree_base(
                        tree, eval_Y, predict(ls_theta, ls_theta1, eval_X), eval_T
                    )
                )

                result = QSA(Xs, Ys, Ts, variant, None, None, config)
                recs["failure_mode"][trial, i] = {
                    "solution_found": 0,
                    "candidate_infeasible": 1,
                    "safety_test_rejected": 2,
                }[result.diagnostics.failure_mode]
                if result.passed_safety:
                    recs["solution_found"][trial, i] = 1.0
                    recs["log_loss"][trial, i] = -float(
                        f_hat(result.theta, result.theta1, eval_X, eval_Y)
                    )
                    recs["g_value"][trial, i] = float(
                        eval_expr_tree_base(
                            tree,
                            eval_Y,
                            predict(result.theta, result.theta1, eval_X),
                            eval_T,
                        )
                    )

        recs["ms"] = np.asarray(ms, dtype=float)
        rows = summarise(recs)
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        plot_all(rows, out_dir)
        save_summary(rows, os.path.join(out_dir, "summary.csv"))
        last = rows[-1]
        print(
            f"{name:22} P(soln)@full={last['p_solution']:.2f} "
            f"P(viol)={last['p_violation']:.2f} "
            f"loss={last['log_loss']:.4f} LSloss={last['ls_log_loss']:.4f} "
            f"LSgap={last['ls_g_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
