"""Minimal end-to-end example: train a fairness-certified classifier.

Runs the Quasi-Seldonian Algorithm (QSA) on synthetic data and reads back the
high-confidence fairness guarantee. This is the script version of the first part
of ``examples/quickstart.ipynb``.

Run it with::

    uv run python examples/quickstart.py
    # or, once the package is installed:  python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np
import torch

from fair_seldonian.algorithms import QSA
from fair_seldonian.data import data_split, get_data
from fair_seldonian.models import eval_ghat, predict


def accuracy(
    theta: torch.Tensor, theta1: torch.Tensor, X: np.ndarray, Y: np.ndarray
) -> float:
    """Fraction of correct predictions at a 0.5 decision threshold."""
    probs = predict(theta, theta1, X).detach().numpy()
    return float(((probs >= 0.5).astype(int) == Y).mean())


def main() -> None:
    # 1. Generate synthetic data. Each row has feature columns (the last of which
    #    is the sensitive attribute T), a binary label Y, and the group label T.
    #    Here both groups have the same base rate (tp0 == tp1), so the data is fair.
    data = get_data(
        N=20000,
        features=5,
        t_ratio=0.5,
        tp0_ratio=0.5,
        tp1_ratio=0.5,
        random_seed=7,
    )

    # 2. Split into test/train arrays: (X_test, Y_test, T_test, X_train, ...).
    X_te, Y_te, T_te, X_tr, Y_tr, T_tr = data_split(
        frac=0.6, all_data=data, random_state=1, m_test=0.3
    )
    print(f"train examples: {X_tr.shape[0]}, test examples: {X_te.shape[0]}")

    # 3. Train with QSA. It returns model parameters and a boolean: either a
    #    model certified to satisfy the fairness constraint with probability
    #    >= 1 - delta, or `passed=False` meaning "No Solution Found".
    theta, theta1, passed = QSA(X_tr, Y_tr, T_tr, "opt", None, None)

    if not passed:
        print("\nNo Solution Found - QSA could not certify a fair model on this data.")
        print("Try more data, a smaller fairness gap, or a larger delta.")
        return

    # 4. The guarantee: eval_ghat returns a high-confidence upper bound on the
    #    constraint function g. A value <= 0 means the constraint is satisfied.
    upper_bound = float(eval_ghat(theta, theta1, X_te, Y_te, T_te, "opt"))
    print("\nModel certified fair.")
    print(f"  fairness upper bound g(theta) <= {upper_bound:.4f}  (<= 0 is satisfied)")
    print(f"  test accuracy: {accuracy(theta, theta1, X_te, Y_te):.3f}")


if __name__ == "__main__":
    main()
