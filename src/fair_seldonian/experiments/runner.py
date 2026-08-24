from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from ..algorithms.qsa import QSA
from ..config import DEFAULT_CONFIG, SeldonianConfig
from ..constraints.expression_tree import construct_expr_tree_base, eval_expr_tree_base
from ..data.synthetic import get_data
from ..models.logistic_regression import f_hat, predict, simple_logistic

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "exp/exp_{}/bin/"

#: Failure-mode codes stored alongside each trial.
FOUND = 0
CANDIDATE_INFEASIBLE = 1
SAFETY_REJECTED = 2
_MODE_CODES = {
    "solution_found": FOUND,
    "candidate_infeasible": CANDIDATE_INFEASIBLE,
    "safety_test_rejected": SAFETY_REJECTED,
}


@dataclass(frozen=True)
class StudySpec:
    """One learning-curve experiment.

    :param seldonian_type: bound-propagation variant (``base``, ``mod``, ``bound``,
        ``const``, ``opt``).
    :param ms: dataset sizes to sweep.
    :param num_trials: independent repetitions at each size. Each repetition draws a
        **fresh** dataset. Generating one dataset and having every "trial"
        subsample it is cheaper but measures subsampling noise rather than sampling
        noise, and at the largest size every trial would see identical data.
    :param force_start: warm-start candidate selection from the previous size's
        solution. This is sound only because each run has its own independent data.
        Under a shared-dataset design the warm start would have been fitted on data
        overlapping the next run's *safety* set, breaking the independence the
        safety test relies on.
    :param eval_size: size of the held-out set used to measure the true constraint
        value and log loss.
    """

    seldonian_type: str = "base"
    ms: tuple[int, ...] = (2_000, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000)
    num_trials: int = 40
    force_start: bool = False
    eval_size: int = 200_000
    config: SeldonianConfig = field(default=DEFAULT_CONFIG)
    t_ratio: float = 0.5
    tp0_ratio: float = 0.4
    tp1_ratio: float = 0.6
    features: int = 5
    seed: int = 0


def true_g(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    config: SeldonianConfig,
) -> float:
    """Plug-in value of the constraint on a held-out set - no confidence interval.

    The Seldonian guarantee is a statement about ``g(theta) <= 0``, so testing it
    means evaluating ``g`` itself. Reaching instead for ``eval_ghat`` - the
    *high-confidence upper bound* - and recording ``1 if eval_ghat(...) > 0`` tests
    a strictly more conservative event, which understates the violation rate and
    leaves the safety claim untested. It also makes the metric saturate: it reads
    zero for every roughly-correct variant and so distinguishes none of them.
    """
    pred = predict(theta, theta1, X)
    tree = construct_expr_tree_base(config.constraint)
    value = eval_expr_tree_base(tree, Y, pred, T)
    return float("nan") if value is None else float(value)


def _run_one(
    spec: StudySpec, trial: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One trial: sweep all dataset sizes, drawing fresh data at each."""
    n_m = len(spec.ms)
    solution_found = np.zeros(n_m)
    log_loss = np.full(n_m, np.nan)
    g_value = np.full(n_m, np.nan)
    mode = np.zeros(n_m)
    ls_log_loss = np.full(n_m, np.nan)
    ls_g_value = np.full(n_m, np.nan)

    # A single held-out set per trial, big enough that its Monte-Carlo error is
    # small next to the effects being measured. Drawn from a disjoint seed range so
    # it can never overlap the training draws.
    eval_data = get_data(
        spec.eval_size,
        spec.features,
        spec.t_ratio,
        spec.tp0_ratio,
        spec.tp1_ratio,
        random_seed=(spec.seed * 1_000_003 + trial + 500_000) / 7919.0,
    )
    eval_X = np.asarray(eval_data.iloc[:, :-2], dtype=float)
    eval_Y = np.asarray(eval_data.iloc[:, -2])
    eval_T = np.asarray(eval_data.iloc[:, -1])

    init_sol: torch.Tensor | None = None
    init_sol1: torch.Tensor | None = None

    for i, m in enumerate(spec.ms):
        data = get_data(
            int(m),
            spec.features,
            spec.t_ratio,
            spec.tp0_ratio,
            spec.tp1_ratio,
            random_seed=(spec.seed * 1_000_003 + trial * 1_009 + i) / 7919.0,
        )
        X = np.asarray(data.iloc[:, :-2], dtype=float)
        Y = np.asarray(data.iloc[:, -2])
        T = np.asarray(data.iloc[:, -1])

        # Unconstrained baseline, fitted on the same data the QSA sees.
        ls_theta, ls_theta1 = simple_logistic(X, Y)
        ls_log_loss[i] = -float(f_hat(ls_theta, ls_theta1, eval_X, eval_Y))
        ls_g_value[i] = true_g(ls_theta, ls_theta1, eval_X, eval_Y, eval_T, spec.config)

        result = QSA(X, Y, T, spec.seldonian_type, init_sol, init_sol1, spec.config)
        mode[i] = _MODE_CODES[result.diagnostics.failure_mode]
        if result.passed_safety:
            solution_found[i] = 1.0
            log_loss[i] = -float(f_hat(result.theta, result.theta1, eval_X, eval_Y))
            g_value[i] = true_g(
                result.theta, result.theta1, eval_X, eval_Y, eval_T, spec.config
            )
            if spec.force_start:
                init_sol, init_sol1 = result.theta, result.theta1

    return solution_found, log_loss, g_value, mode, ls_log_loss, ls_g_value


def run_study(
    spec: StudySpec, output_file: str | None = None, n_jobs: int = 1
) -> dict[str, np.ndarray]:
    """Run every trial of ``spec`` and return the raw per-trial records.

    Nothing is averaged here. "No solution found" is recorded as ``NaN`` for loss
    and constraint value rather than 0, so that it cannot be silently averaged in.
    Storing 0 instead produces a dip-and-return shape in the constraint panel that
    reads as though found solutions sit comfortably below the threshold, when it is
    really an artifact of mixing refusals into the mean.
    """
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            rows = list(
                pool.map(_run_one, [spec] * spec.num_trials, range(spec.num_trials))
            )
    else:
        rows = [_run_one(spec, t) for t in range(spec.num_trials)]

    out = {
        "ms": np.asarray(spec.ms, dtype=float),
        "solution_found": np.vstack([r[0] for r in rows]),
        "log_loss": np.vstack([r[1] for r in rows]),
        "g_value": np.vstack([r[2] for r in rows]),
        "failure_mode": np.vstack([r[3] for r in rows]),
        "ls_log_loss": np.vstack([r[4] for r in rows]),
        "ls_g_value": np.vstack([r[5] for r in rows]),
    }
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        np.savez(output_file, **out)
        logger.info(f"Saved {output_file}")
    return out
