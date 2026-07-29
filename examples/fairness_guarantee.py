"""Contrast a fair dataset (certified) with an unfair one (No Solution Found).

The Seldonian guarantee is one-sided: QSA will only return a model when it can
certify the fairness constraint holds with high probability. On data with a large
group disparity it returns "No Solution Found" rather than an unsafe model.

Run it with::

    uv run python examples/fairness_guarantee.py
"""

from __future__ import annotations

from fair_seldonian.algorithms import QSA
from fair_seldonian.data import data_split, get_data
from fair_seldonian.models import eval_ghat


def run(label: str, tp0_ratio: float, tp1_ratio: float) -> None:
    """Train on data whose two groups have base rates tp0_ratio and tp1_ratio."""
    data = get_data(
        N=20000,
        features=5,
        t_ratio=0.5,
        tp0_ratio=tp0_ratio,
        tp1_ratio=tp1_ratio,
        random_seed=7,
    )
    X_te, Y_te, T_te, X_tr, Y_tr, T_tr = data_split(
        frac=0.6, all_data=data, random_state=1, m_test=0.3
    )
    theta, theta1, passed = QSA(X_tr, Y_tr, T_tr, "opt", None, None)

    gap = abs(tp1_ratio - tp0_ratio)
    print(f"{label} (group base-rate gap = {gap:.2f}):")
    if passed:
        ub = float(eval_ghat(theta, theta1, X_te, Y_te, T_te, "opt"))
        print(f"  -> certified fair; fairness upper bound <= {ub:.4f}\n")
    else:
        print("  -> No Solution Found; QSA refused to return an unfair model\n")


def main() -> None:
    run("Fair data", tp0_ratio=0.5, tp1_ratio=0.5)
    run("Unfair data", tp0_ratio=0.2, tp1_ratio=0.8)


if __name__ == "__main__":
    main()
