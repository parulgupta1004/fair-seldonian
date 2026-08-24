from __future__ import annotations

import os
from typing import Any

#: Panel filenames, kept as-is so existing figure references keep resolving.
LOSS_PANEL = "tutorial7MSE_py.png"
VIOLATION_PANEL = "tutorial7PrFail1_py.png"
CONSTRAINT_PANEL = "tutorial7PrFail2_py.png"
SOLUTION_PANEL = "tutorial7PrSoln_py.png"


def _pyplot() -> Any:
    """Import pyplot on first use rather than at module import.

    matplotlib is an optional extra (``plots``, ``notebook``), but
    ``experiments/__init__`` re-exports :func:`plot_all`. Importing it eagerly
    therefore made the whole subpackage unimportable without matplotlib -
    including ``summarise``, ``run_study`` and ``clopper_pearson``, none of
    which draw anything. A base install, or CI installing only the ``dev``
    extra, hit ``ModuleNotFoundError`` on ``import fair_seldonian.experiments``.

    Agg is selected before pyplot loads so this works with no display attached.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(ax, xlabel: str, ylabel: str, out_path: str, legend_loc: str) -> None:
    plt = _pyplot()
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xscale("log")
    ax.legend(loc=legend_loc, fontsize=11)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close()


def plot_all(rows: list[dict[str, float]], out_dir: str) -> None:
    """Write the four result panels for one experiment variant."""
    plt = _pyplot()
    ms = [r["m"] for r in rows]

    # Log loss. Only trials that returned a solution contribute, so points backed by
    # too few solutions are dropped rather than drawn as if they were comparable.
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    shown = [(r["m"], r) for r in rows if r["n_solutions"] >= 5]
    if shown:
        ax.errorbar(
            [m for m, _ in shown],
            [r["log_loss"] for _, r in shown],
            yerr=[r["log_loss_se"] for _, r in shown],
            fmt="o-",
            color="tab:blue",
            capsize=3,
            linewidth=2,
            label="QSA",
        )
    ax.errorbar(
        ms,
        [r["ls_log_loss"] for r in rows],
        yerr=[r["ls_log_loss_se"] for r in rows],
        fmt="s-",
        color="tab:red",
        capsize=3,
        linewidth=2,
        label="Logistic regression",
    )
    _finish(
        ax,
        "Training set size",
        "Log loss (held-out)",
        os.path.join(out_dir, LOSS_PANEL),
        "best",
    )

    # Probability that the returned solution actually violates the constraint.
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.errorbar(
        ms,
        [r["p_violation"] for r in rows],
        yerr=[
            [r["p_violation"] - r["p_violation_lo"] for r in rows],
            [r["p_violation_hi"] - r["p_violation"] for r in rows],
        ],
        fmt="o-",
        color="tab:blue",
        capsize=3,
        linewidth=2,
        label="QSA",
    )
    ax.errorbar(
        ms,
        [float(r["ls_g_mean"] > 0) for r in rows],
        fmt="s-",
        color="tab:red",
        linewidth=2,
        label="Logistic regression",
    )
    ax.axhline(0.05, ls="--", color="0.4", label=r"$\delta = 0.05$")
    ax.set_ylim(-0.05, 1.08)
    _finish(
        ax,
        "Training set size",
        r"P(true $g(\theta) > 0$)",
        os.path.join(out_dir, VIOLATION_PANEL),
        "best",
    )

    # True constraint value of the returned solutions.
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    if shown:
        ax.errorbar(
            [m for m, _ in shown],
            [r["g_mean"] for _, r in shown],
            yerr=[r["g_se"] for _, r in shown],
            fmt="o-",
            color="tab:blue",
            capsize=3,
            linewidth=2,
            label="QSA",
        )
    ax.plot(
        ms,
        [r["ls_g_mean"] for r in rows],
        "s-",
        color="tab:red",
        linewidth=2,
        label="Logistic regression",
    )
    ax.axhline(0.0, ls="--", color="0.4")
    _finish(
        ax,
        "Training set size",
        r"True $g(\theta)$",
        os.path.join(out_dir, CONSTRAINT_PANEL),
        "best",
    )

    # Probability of returning any solution, with the failure split underneath.
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.errorbar(
        ms,
        [r["p_solution"] for r in rows],
        yerr=[
            [r["p_solution"] - r["p_solution_lo"] for r in rows],
            [r["p_solution_hi"] - r["p_solution"] for r in rows],
        ],
        fmt="o-",
        color="tab:blue",
        capsize=3,
        linewidth=2,
        label="QSA",
    )
    ax.plot(
        ms,
        [r["frac_candidate_infeasible"] for r in rows],
        "^--",
        color="tab:orange",
        linewidth=1.4,
        label="failed: candidate infeasible",
    )
    ax.plot(
        ms,
        [r["frac_safety_rejected"] for r in rows],
        "v--",
        color="tab:green",
        linewidth=1.4,
        label="failed: safety test",
    )
    ax.set_ylim(-0.05, 1.08)
    _finish(
        ax,
        "Training set size",
        "Probability of a solution",
        os.path.join(out_dir, SOLUTION_PANEL),
        "best",
    )
