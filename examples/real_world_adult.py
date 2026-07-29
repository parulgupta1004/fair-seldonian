"""Apply QSA to a real dataset: UCI Adult income.

This example downloads the Adult dataset (via scikit-learn / OpenML), frames a
fairness task, and shows the two sides of the Seldonian guarantee on real data:

* An ordinary logistic regression achieves good accuracy but has a measurable
  true-positive-rate gap between demographic groups.
* QSA, asked to certify that gap is bounded, returns "No Solution Found" on this
  data rather than shipping the biased model.

Task framing:
    label Y     = income > 50K
    sensitive T = sex (1 = Male, 0 = Female)
    features X  = standardized numeric columns, with T appended as the final
                  column (the convention used throughout this library).

Requires network access on first run (the dataset is cached afterwards). Run::

    uv run python examples/real_world_adult.py
"""

from __future__ import annotations

import numpy as np

from fair_seldonian.algorithms import QSA
from fair_seldonian.models import predict, simple_logistic


def load_adult() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, T) arrays for the Adult dataset, or raise on no network."""
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(
        "adult", version=2, as_frame=True, parser="auto"
    ).frame.dropna()
    T = (frame["sex"].astype(str) == "Male").astype(int).to_numpy()
    Y = frame["class"].astype(str).str.contains(">50K").astype(int).to_numpy()

    numeric = frame.select_dtypes("number")
    standardized = (numeric - numeric.mean()) / numeric.std()
    # Append the sensitive attribute as the final feature column (library convention).
    X = np.column_stack([standardized.to_numpy(), T]).astype(float)
    return X, Y, T


def true_positive_rate(pred: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> float:
    """P(pred = 1 | Y = 1) within the group selected by `mask`."""
    positives = mask & (Y == 1)
    if not positives.any():
        return float("nan")
    return float(pred[positives].mean())


def main() -> None:
    try:
        X, Y, T = load_adult()
    except (
        Exception
    ) as exc:  # network/OpenML errors - keep the example runnable offline
        print(f"Could not load the Adult dataset ({type(exc).__name__}: {exc}).")
        print("This example needs network access on first run to download it.")
        return

    # Deterministic subsample + split for a fast, reproducible demo.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))[:8000]
    X, Y, T = X[idx], Y[idx], T[idx]
    cut = int(0.7 * len(X))
    X_tr, Y_tr, T_tr = X[:cut], Y[:cut], T[:cut]
    X_te, Y_te, T_te = X[cut:], Y[cut:], T[cut:]
    print(
        f"Adult: {len(X)} examples, positive rate {Y.mean():.3f}, "
        f"male share {T.mean():.3f}"
    )

    # 1. Unconstrained baseline: standard logistic regression.
    theta, theta1 = simple_logistic(X_tr, Y_tr)
    pred = (predict(theta, theta1, X_te).detach().numpy() >= 0.5).astype(int)
    acc = float((pred == Y_te).mean())
    tpr_male = true_positive_rate(pred, Y_te, T_te == 1)
    tpr_female = true_positive_rate(pred, Y_te, T_te == 0)
    print("\nUnconstrained logistic regression:")
    print(f"  accuracy            : {acc:.3f}")
    print(f"  TPR (male)          : {tpr_male:.3f}")
    print(f"  TPR (female)        : {tpr_female:.3f}")
    gap = abs(tpr_male - tpr_female)
    print(f"  TPR gap |M - F|     : {gap:.3f}   <- the bias QSA guards against")

    # 2. QSA with the default equalized-opportunity constraint.
    _, _, passed = QSA(X_tr, Y_tr, T_tr, "opt", None, None)
    print("\nQuasi-Seldonian Algorithm:")
    if passed:
        print("  -> certified fair")
    else:
        print("  -> No Solution Found: QSA will not certify a model on this data,")
        print("     rather than return one with the disparity shown above.")


if __name__ == "__main__":
    main()
