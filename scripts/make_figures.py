"""Generate the two analysis figures that are functions rather than tables.

Both are computed here rather than drawn by hand, so re-running this after any
change to the bound machinery keeps the paper's figures honest.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fair_seldonian.config import SeldonianConfig  # noqa: E402
from fair_seldonian.constraints.expression_tree import (  # noqa: E402
    construct_expr_tree_base,
    eval_expr_tree_base,
)
from fair_seldonian.constraints.fairness import equal_opportunity  # noqa: E402
from fair_seldonian.data.synthetic import get_data  # noqa: E402
from fair_seldonian.models.logistic_regression import (  # noqa: E402
    eval_ghat,
    predict,
    simple_logistic,
)


def half_widths(p: np.ndarray, n: float, delta: float) -> tuple[np.ndarray, float]:
    """Two-sided empirical-Bernstein and Hoeffding half-widths at rate ``p``."""
    log_term = math.log(2 / delta)
    hoeffding = math.sqrt(log_term / (2 * n))
    variance = p * (1 - p)
    bernstein = np.sqrt(2 * variance * log_term / n) + 7 * log_term / (3 * (n - 1))
    return bernstein, hoeffding


def operating_point_figure(path: str, delta: float = 0.05) -> None:
    """Where empirical Bernstein beats Hoeffding, as a function of the rate.

    Hoeffding assumes the worst-case variance of 1/4, which is tight only at
    p = 1/2. Everything to the left of the crossing is where paying for measured
    variance is worth something.
    """
    p = np.linspace(0.02, 0.5, 400)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))

    # The two datasets in this paper, drawn first so the curves sit on top.
    ax.axvspan(0.28, 0.50, color="tab:red", alpha=0.09)
    ax.axvline(0.114, color="tab:green", linewidth=1.4)

    for n, style, width, label in [
        (10_000, ":", 1.2, "$n = 10^4$"),
        (120_000, "-", 2.2, "$n = 10^5$"),
        (1_000_000, "--", 1.2, "$n = 10^6$"),
    ]:
        bern, hoef = half_widths(p, n, delta)
        ax.plot(p, bern / hoef, style, color="tab:blue", linewidth=width, label=label)

    ax.axhline(1.0, color="0.4", linewidth=1.0)
    ax.text(0.20, 1.030, "Bernstein wider", fontsize=8.5, color="0.4", va="bottom")
    ax.text(0.20, 0.970, "Bernstein narrower", fontsize=8.5, color="0.4", va="top")

    ax.text(
        0.39,
        1.105,
        "synthetic benchmark",
        ha="center",
        fontsize=8.5,
        color="tab:red",
        va="top",
    )
    ax.text(
        0.104,
        1.105,
        "Adult (women)",
        ha="left",
        fontsize=8.5,
        color="tab:green",
        va="top",
    )

    ax.set_xlabel("group rate $p$", fontsize=11)
    ax.set_ylabel("interval width,\nBernstein $\\div$ Hoeffding", fontsize=11)
    ax.set_xlim(0.5, 0.02)  # left to right = further from 1/2
    ax.set_ylim(0.35, 1.15)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def certification_gap_figure(
    path: str, summary_csv: str, epsilon: float = 0.10, delta: float = 0.05
) -> None:
    """What it costs to prove a constraint the model already satisfies.

    Under equal opportunity at this epsilon the unconstrained classifier is
    already fair, but its bound stays above zero: certifying *that* model would
    need orders of magnitude more data. QSA answers much sooner by returning a
    different, heavily over-corrected model instead. The distance between those
    two answers is the slack, priced in data.
    """
    import csv

    constraint = equal_opportunity(epsilon)
    config = SeldonianConfig(constraint=constraint, delta=delta, candidate_ratio=0.60)
    tree = construct_expr_tree_base(constraint)

    evaluation = get_data(200_000, 5, 0.5, 0.4, 0.6, random_seed=0.77)
    eval_X = np.asarray(evaluation.iloc[:, :-2], dtype=float)
    eval_Y = np.asarray(evaluation.iloc[:, -2])
    eval_T = np.asarray(evaluation.iloc[:, -1])

    sizes = [10_000, 20_000, 40_000, 80_000, 160_000, 320_000, 640_000]
    true_g, bound = [], []
    for n in sizes:
        data = get_data(n, 5, 0.5, 0.4, 0.6, random_seed=(n % 977) / 313.0)
        X = np.asarray(data.iloc[:, :-2], dtype=float)
        Y = np.asarray(data.iloc[:, -2])
        T = np.asarray(data.iloc[:, -1])
        theta, theta1 = simple_logistic(X, Y)
        true_g.append(
            float(
                eval_expr_tree_base(
                    tree, eval_Y, predict(theta, theta1, eval_X), eval_T
                )
            )
        )
        bound.append(float(eval_ghat(theta, theta1, X, Y, T, "base", config)))

    # What QSA actually returns, from the measured run.
    qsa = {int(float(r["m"])): r for r in csv.DictReader(open(summary_csv))}
    qsa_n = [n for n in sizes if n in qsa and qsa[n]["g_mean"] not in ("", "nan")]
    qsa_g = [float(qsa[n]["g_mean"]) for n in qsa_n]
    first_certified = min(qsa_n) if qsa_n else None

    # Slack falls as 1/sqrt(n); extrapolate to where it clears |g|.
    needed = sizes[-1] * (bound[-1] - true_g[-1]) ** 2 / true_g[-1] ** 2

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.axhline(0.0, color="0.4", linewidth=1.0)
    ax.fill_between(sizes, true_g, bound, color="tab:blue", alpha=0.09)

    ax.plot(
        sizes,
        bound,
        "s-",
        color="tab:blue",
        linewidth=2,
        label="bound on the plain model",
    )
    ax.plot(
        sizes,
        true_g,
        "o-",
        color="tab:green",
        linewidth=2,
        label="true value, plain model",
    )
    ax.plot(
        qsa_n,
        qsa_g,
        "^-",
        color="tab:orange",
        linewidth=2,
        label="true value, model QSA returns",
    )

    ax.text(1.4e4, (bound[1] + true_g[1]) / 2, "slack", fontsize=9, color="tab:blue")
    ax.text(
        1.05e4, 0.012, "fails to certify above this line", fontsize=8.5, color="0.4"
    )

    if first_certified:
        ax.axvline(first_certified, color="0.5", linestyle=":", linewidth=1.3)
        ax.annotate(
            f"QSA certifies\nfrom $n={first_certified // 1000}$k",
            xy=(first_certified, -0.06),
            xytext=(7e4, -0.055),
            fontsize=9,
            color="0.3",
            arrowprops=dict(arrowstyle="->", color="0.5", linewidth=1),
        )

    ax.set_xscale("log")
    ax.set_xlabel("training set size", fontsize=11)
    ax.set_ylabel("equal-opportunity constraint $g$", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {path} | plain model needs n ~ {needed:,.0f} to certify; "
        f"QSA certifies from {first_certified}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    operating_point_figure(os.path.join(args.out, "fig_operating_point.png"))
    certification_gap_figure(
        os.path.join(args.out, "fig_certification_gap.png"),
        os.path.join(args.out, "eo", "base", "summary.csv"),
    )
