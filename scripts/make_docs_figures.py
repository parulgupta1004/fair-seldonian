"""Draw the documentation's two explanatory figures from the library itself.

Both illustrate a claim the prose makes, so both are computed by calling the
real implementations rather than sketched: re-running this after a change to the
bound machinery is what keeps the figures honest. ``tests/test_docs_sync.py``
re-runs it and fails if the committed SVGs no longer match, on the same
flag-don't-rewrite contract as the notebook freshness check.

Run it with::

    uv run python scripts/make_docs_figures.py

Colours are the first four slots of the validated categorical palette, in fixed
order. Two of the four fall below 3:1 against the surface, so identity never
rests on hue: the line chart carries a colour-matched legend and the same
numbers appear as a table on the page beside it.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fair_seldonian import (  # noqa: E402
    DEFAULT_CONFIG,
    data_split,
    eval_ghat,
    get_data,
    simple_logistic,
)
from fair_seldonian.constraints.expression_tree import (  # noqa: E402
    construct_expr_tree_base,
    eval_expr_tree_base,
)
from fair_seldonian.constraints.inequalities import (  # noqa: E402
    betting_interval,
    eval_empirical_bernstein,
    eval_hoeffding,
    eval_t_test,
)
from fair_seldonian.models.logistic_regression import SELDONIAN_TYPES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "_static" / "generated"

#: Validated categorical slots 1-4, assigned in fixed order and never cycled.
SERIES = {
    "HOEFFDING_INEQUALITY": "#2a78d6",
    "EMPIRICAL_BERNSTEIN": "#eb6834",
    "BETTING": "#1baf7a",
    "T_TEST": "#eda100",
}
INK = "#1b1b1d"
MUTED = "#5c5c5f"
GRID = "#e3e3e0"
SURFACE = "#ffffff"

DELTA = 0.05

#: Draws averaged per sample size, enough to settle the adaptive intervals.
REPLICATES = 15


def _style(ax) -> None:
    """Recessive axes and grid: the data should be the darkest thing on the page."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(MUTED)


def _half_widths(x: np.ndarray) -> dict[str, float]:
    """Each inequality's two-sided half-width for one sample of [0, 1] values."""
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    n = x.size
    _, hi_bet = betting_interval(x, DELTA, True)
    return {
        "HOEFFDING_INEQUALITY": eval_hoeffding(mean, n, DELTA, True)[1] - mean,
        "EMPIRICAL_BERNSTEIN": eval_empirical_bernstein(mean, var, n, DELTA, True)[1]
        - mean,
        "BETTING": hi_bet - mean,
        "T_TEST": eval_t_test(mean, var**0.5, n, DELTA, True)[1] - mean,
    }


def half_width_curves(sizes, rate: float, replicates: int) -> dict[str, list[float]]:
    """Mean half-width per inequality at each sample size.

    Averaged over draws: a single draw per size makes the betting curve visibly
    jagged -- it adapts to the sample, so its width moves with the sample
    variance -- and a reader reads that noise as the method being erratic.
    """
    curves: dict[str, list[float]] = {k: [] for k in SERIES}
    for n in sizes:
        runs: dict[str, list[float]] = {k: [] for k in SERIES}
        for seed in range(replicates):
            rng = np.random.default_rng(1000 * seed + int(n))
            x = rng.binomial(1, rate, int(n)).astype(float)
            for name, width in _half_widths(x).items():
                runs[name].append(width)
        for name, widths in runs.items():
            curves[name].append(float(np.mean(widths)))
    return curves


def interval_width_figure(path: Path) -> None:
    """Half-width against sample size, at a small rate and at the worst case.

    Two panels because the interesting part is that the ordering changes. At a
    base rate of 0.1 empirical Bernstein is well inside Hoeffding; at 0.5 it is
    slightly *outside* it, having paid the additive variance-estimation term for
    a variance that is already the worst case.
    """
    sizes = np.array([250, 500, 1000, 2000, 4000, 8000, 16000, 32000])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), facecolor=SURFACE, sharey=True)

    for ax, rate in zip(axes, (0.1, 0.5)):
        curves = half_width_curves(sizes, rate, REPLICATES)
        for name, colour in SERIES.items():
            ax.plot(
                sizes,
                curves[name],
                color=colour,
                linewidth=2,
                zorder=3,
                label=name.replace("_INEQUALITY", "").replace("_", " ").title(),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("sample size $n$", fontsize=9, color=MUTED)
        ax.set_title(
            f"base rate {rate}"
            + ("  (worst case for the variance)" if rate == 0.5 else ""),
            fontsize=9.5,
            color=INK,
            pad=8,
        )
        _style(ax)
    axes[0].set_ylabel(
        f"interval half-width  ($\\delta$={DELTA})", fontsize=9, color=MUTED
    )
    # A legend rather than direct labels: the curves converge at the right-hand
    # edge, where four labels would sit on top of one another. Identity is still
    # never carried by hue alone, and the same numbers appear as a table on the
    # page, which is the relief the contrast check asks for.
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        frameon=False,
        fontsize=8.5,
    )
    for text, colour in zip(legend.get_texts(), SERIES.values()):
        text.set_color(colour)
    fig.savefig(
        path,
        format=path.suffix[1:],
        bbox_inches="tight",
        facecolor=SURFACE,
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)


def variant_slack() -> tuple[list[str], list[float], int]:
    """``(variant names, slack over the true value, rows used)``.

    Drawn on the default constraint rather than demographic parity because that
    one exercises every variant: it repeats ``TP(1)``, which is what ``bound``
    merges, and carries a literal ``0.25``, which is what ``const`` exploits. On
    demographic parity both are no-ops and three bars come out identical.
    """
    from fair_seldonian.models.logistic_regression import predict

    data = get_data(
        N=20000, features=5, t_ratio=0.5, tp0_ratio=0.4, tp1_ratio=0.6, random_seed=0
    )
    _, _, _, X, Y, T = data_split(frac=0.6, all_data=data, random_state=0, m_test=0.2)
    theta, theta1 = simple_logistic(X, Y)
    theta, theta1 = theta.detach(), theta1.detach()
    config = DEFAULT_CONFIG
    truth = float(
        eval_expr_tree_base(
            construct_expr_tree_base(config.constraint), Y, predict(theta, theta1, X), T
        )
    )
    names = list(SELDONIAN_TYPES)
    slack = [
        float(eval_ghat(theta, theta1, X, Y, T, name, config)) - truth for name in names
    ]
    return names, slack, len(X)


def variant_slack_figure(path: Path) -> None:
    """How much each ``seldonian_type`` overstates the constraint it bounds.

    Slack is the gap between the bound and the true value on the same data and
    the same model: the smaller the bar, the less the variant gives away.

    Drawn on the default constraint rather than demographic parity, because that
    one exercises every variant: it repeats ``TP(1)``, which is what ``bound``
    merges, and it carries a literal ``0.25``, which is what ``const`` exploits.
    On demographic parity both are no-ops and three bars come out identical.

    ``mod`` still coincides with ``base`` here, and that is not a bug -- it
    changes the *predicted* bound that steers candidate selection, not the
    safety bound being measured.
    """
    names, slack, n_rows = variant_slack()
    config = DEFAULT_CONFIG

    fig, ax = plt.subplots(figsize=(6.6, 3.2), facecolor=SURFACE)
    ypos = np.arange(len(names))[::-1]
    # One series, so one hue; the title names what is measured and no legend is
    # needed. 4px rounded ends anchored at the baseline.
    bars = ax.barh(
        ypos, slack, height=0.62, color=SERIES["HOEFFDING_INEQUALITY"], zorder=3
    )
    for bar, value in zip(bars, slack):
        ax.annotate(
            f"{value:.4f}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    ax.set_yticks(ypos, names, fontsize=9)
    ax.set_xlabel(
        f"slack over the true value  ({config.constraint!r}, "
        f"$\\delta$={config.delta}, n={n_rows:,})",
        fontsize=9,
        color=MUTED,
    )
    ax.set_xlim(0, max(slack) * 1.18)
    _style(ax)
    for label in ax.get_yticklabels():
        label.set_color(INK)
        label.set_fontfamily("monospace")
    fig.savefig(path, format=path.suffix[1:], bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


FIGURES = {
    "interval-widths": interval_width_figure,
    "variant-slack": variant_slack_figure,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR), help="output directory")
    parser.add_argument(
        "--format", default="svg", choices=["svg", "png"], help="image format"
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        path = out / f"{name}.{args.format}"
        draw(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
