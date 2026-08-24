from __future__ import annotations

import csv
import logging
import os

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval.

    Probabilities here are estimated from a few dozen trials, where the normal
    approximation is poor and - at the boundary, where a rate of exactly 0 or 1 is
    common - gives a zero-width interval. Reporting "violation rate 0.00" from 40
    trials as though it established anything about a 0.05 threshold is the specific
    error this avoids: 0/40 is consistent with a true rate up to about 0.088.
    """
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(alpha / 2, successes, n - successes + 1))
    )
    upper = (
        1.0
        if successes == n
        else float(stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    )
    return lower, upper


def _mean_stderr(values: np.ndarray) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)]
    n = finite.size
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(finite[0]), float("nan"), 1
    return float(finite.mean()), float(finite.std(ddof=1) / np.sqrt(n)), n


def summarise(records: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """Collapse per-trial records into one row per dataset size.

    Two aggregation choices here are deliberate and easy to get wrong.

    Loss and constraint value are averaged only over trials that returned a
    solution, and the number of such trials is reported alongside. They are
    conditional means, and the conditioning set changes along the x-axis - at one
    end almost every trial contributes, at the other almost none. Comparing two
    variants' loss curves without knowing how many trials sit behind each point is
    not meaningful, so ``n_solutions`` travels with the number.

    The violation rate counts trials where a solution was returned *and* its true
    constraint value is positive. A run that returns "no solution found" has not
    violated anything - it has declined to answer - so it is excluded from the
    numerator but kept in the denominator, which is the event the guarantee
    actually bounds.
    """
    ms = records["ms"]
    rows = []
    for i, m in enumerate(ms):
        found = records["solution_found"][:, i]
        n_trials = int(found.size)
        n_found = int(found.sum())
        p_lo, p_hi = clopper_pearson(n_found, n_trials)

        g = records["g_value"][:, i]
        violations = int(np.sum(np.isfinite(g) & (g > 0)))
        v_lo, v_hi = clopper_pearson(violations, n_trials)

        loss_mean, loss_se, loss_n = _mean_stderr(records["log_loss"][:, i])
        g_mean, g_se, _ = _mean_stderr(g)
        ls_loss_mean, ls_loss_se, _ = _mean_stderr(records["ls_log_loss"][:, i])
        ls_g_mean, _, _ = _mean_stderr(records["ls_g_value"][:, i])

        modes = records["failure_mode"][:, i]
        rows.append(
            {
                "m": float(m),
                "n_trials": n_trials,
                "p_solution": n_found / n_trials,
                "p_solution_lo": p_lo,
                "p_solution_hi": p_hi,
                "p_violation": violations / n_trials,
                "p_violation_lo": v_lo,
                "p_violation_hi": v_hi,
                "log_loss": loss_mean,
                "log_loss_se": loss_se,
                "n_solutions": loss_n,
                "g_mean": g_mean,
                "g_se": g_se,
                "ls_log_loss": ls_loss_mean,
                "ls_log_loss_se": ls_loss_se,
                "ls_g_mean": ls_g_mean,
                "frac_candidate_infeasible": float(np.mean(modes == 1)),
                "frac_safety_rejected": float(np.mean(modes == 2)),
            }
        )
    return rows


FIELDS = [
    "m",
    "n_trials",
    "p_solution",
    "p_solution_lo",
    "p_solution_hi",
    "p_violation",
    "p_violation_lo",
    "p_violation_hi",
    "log_loss",
    "log_loss_se",
    "n_solutions",
    "g_mean",
    "g_se",
    "ls_log_loss",
    "ls_log_loss_se",
    "ls_g_mean",
    "frac_candidate_infeasible",
    "frac_safety_rejected",
]


def save_summary(rows: list[dict[str, float]], filename: str) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})
    logger.info(f"Wrote {filename}")
