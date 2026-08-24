from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import torch
from scipy.optimize import minimize

from ..config import DEFAULT_CONFIG, SeldonianConfig
from ..constraints.expression_tree import constraint_groups
from ..constraints.inequalities import check_constraint_groups
from ..models.logistic_regression import eval_ghat, f_hat, ghat, simple_logistic

#: The names this module contributes to the public API. autodoc documents
#: exactly these, so the API reference stays the surface users are meant to
#: call rather than every helper that happens to lack a leading underscore.
__all__ = [
    "Diagnostics",
    "QSA",
    "QSAResult",
    "get_cand_solution",
    "safety_test",
    "split_candidate_safety",
]

logger = logging.getLogger(__name__)


def passes_safety(upper_bound: float) -> bool:
    """The safety test's decision rule.

    Defined once because three places need it - :func:`QSA`, :func:`safety_test`
    and :attr:`Diagnostics.failure_mode` - and a criterion that drifts between
    them would report a model as certified in one and rejected in another.
    """
    return bool(upper_bound <= 0.0)


class Diagnostics(NamedTuple):
    """Why a run ended the way it did.

    ``candidate_upper_bound > 0`` means candidate selection never reached a
    feasible point and the safety test was doomed before it ran; that is a
    *different* failure from a feasible candidate being rejected on the safety
    data, and the two call for different remedies. Reporting only "no solution
    found" conflates them, and a non-monotone solution curve then invites an
    explanation in terms of the bound or the iteration budget when the real cause
    is that candidate selection never left the infeasible region.
    """

    candidate_upper_bound: float
    safety_upper_bound: float
    optimizer_status: int
    optimizer_iterations: int
    optimizer_evaluations: int
    optimizer_message: str

    @property
    def failure_mode(self) -> str:
        if passes_safety(self.safety_upper_bound):
            return "solution_found"
        if self.candidate_upper_bound > 0.0:
            return "candidate_infeasible"
        return "safety_test_rejected"


class QSAResult(NamedTuple):
    theta: torch.Tensor
    theta1: torch.Tensor
    passed_safety: bool
    diagnostics: Diagnostics


def split_candidate_safety(
    X: np.ndarray, Y: np.ndarray, T: np.ndarray, candidate_ratio: float
) -> tuple[np.ndarray, ...]:
    """Split into candidate and safety sets at a single, shared index.

    Splitting ``X``/``Y`` and ``T`` through two different code paths - say
    ``train_test_split(test_size=1 - candidate_ratio)`` for one and ``np.split``
    at ``int(candidate_ratio * n)`` for the other - relies on two rounding rules
    agreeing, and they round opposite ways. They disagree by a row at some
    ratios (0.7 and 0.44 among them, though not the 0.4 default), which leaves
    ``T`` a different length from the predictions and raises ``IndexError`` from
    the group mask. Computing the boundary once removes the possibility.

    Rows are *not* shuffled here - callers are responsible for supplying data in
    exchangeable order. On a dataset with meaningful row order (UCI Adult is not
    shuffled) an unshuffled split makes the candidate and safety sets
    non-exchangeable, which breaks the i.i.d. premise the guarantee rests on.
    """
    n_candidate = int(round(candidate_ratio * len(X)))
    return (
        X[:n_candidate],
        X[n_candidate:],
        Y[:n_candidate],
        Y[n_candidate:],
        T[:n_candidate],
        T[n_candidate:],
    )


def QSA(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    seldonian_type: str,
    init_sol: torch.Tensor | None,
    init_sol1: torch.Tensor | None,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> QSAResult:
    """
    Run the quasi-Seldonian algorithm.

    :param X: The features of the dataset
    :param Y: The corresponding labels of the dataset
    :param T: The corresponding sensitive attributes of the dataset
    :param seldonian_type: The mode used in the experiment
    :param init_sol: Initial theta values for the model. Must not depend on data
        that ends up in the safety split - the guarantee requires the candidate
        solution to be independent of the safety set.
    :param init_sol1: The additional initial theta values for the model
    :param config: Algorithm configuration
    :return: :class:`QSAResult`
    """
    # Fail loudly on a group the constraint names but T does not contain. Without
    # this the masks come back empty, the bound fails closed to +inf and the run
    # reports an ordinary "no solution found" instead of the type error it is.
    check_constraint_groups(constraint_groups(config.constraint), T)

    cand_X, safe_X, cand_Y, safe_Y, cand_T, safe_T = split_candidate_safety(
        X, Y, T, config.candidate_ratio
    )

    theta, theta1, opt = get_cand_solution(
        cand_X, cand_Y, cand_T, seldonian_type, init_sol, init_sol1, config
    )

    candidate_upper_bound = float(
        eval_ghat(theta, theta1, cand_X, cand_Y, cand_T, seldonian_type, config)
    )
    safety_upper_bound = float(
        eval_ghat(theta, theta1, safe_X, safe_Y, safe_T, seldonian_type, config)
    )
    diagnostics = Diagnostics(
        candidate_upper_bound=candidate_upper_bound,
        safety_upper_bound=safety_upper_bound,
        optimizer_status=int(getattr(opt, "status", -1)),
        optimizer_iterations=int(getattr(opt, "nit", -1)),
        optimizer_evaluations=int(getattr(opt, "nfev", -1)),
        optimizer_message=str(getattr(opt, "message", "")),
    )
    logger.debug(
        "candidate u=%.6f safety u=%.6f mode=%s optimizer=%s nit=%d nfev=%d",
        candidate_upper_bound,
        safety_upper_bound,
        diagnostics.failure_mode,
        diagnostics.optimizer_message,
        diagnostics.optimizer_iterations,
        diagnostics.optimizer_evaluations,
    )
    return QSAResult(theta, theta1, passes_safety(safety_upper_bound), diagnostics)


def safety_test(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    safe_data_X: np.ndarray,
    safe_data_Y: np.ndarray,
    safe_data_T: np.ndarray,
    seldonian_type: str,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> bool:
    """
    This function does the safety test.

    :param theta: The optimal theta values for the model
    :param theta1: The additional optimal theta values for the model
    :param safe_data_X: The features of the safety dataset
    :param safe_data_Y: The corresponding labels of the safety dataset
    :param safe_data_T: The corresponding sensitive attributes of the safety dataset
    :param seldonian_type: The mode used in the experiment
    :param config: Algorithm configuration
    :return: Whether the candidate solution passed the safety test.
    """
    upper_bound = eval_ghat(
        theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type, config
    )
    logger.debug(f"Safety test upperbound: {upper_bound}")
    return passes_safety(float(upper_bound))


def get_cand_solution(
    cand_data_X: np.ndarray,
    cand_data_Y: np.ndarray,
    cand_data_T: np.ndarray,
    seldonian_type: str,
    init_sol: torch.Tensor | None,
    init_sol1: torch.Tensor | None,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """
    This function provides the candidate solution.

    :return: ``(theta, theta1, optimizer_result)``.
    """
    if init_sol is None or init_sol1 is None:
        init_sol, init_sol1 = simple_logistic(cand_data_X, cand_data_Y)
    init_theta = np.concatenate((init_sol.detach().numpy(), init_sol1.detach().numpy()))
    res = minimize(
        cand_obj,
        x0=init_theta,
        method=config.optimizer,
        options={"disp": False, "maxiter": config.max_iter},
        args=(cand_data_X, cand_data_Y, cand_data_T, seldonian_type, config),
    )
    theta = torch.tensor(np.atleast_1d(res.x)[:-1])
    theta1 = torch.tensor(np.array([np.atleast_1d(res.x)[-1]]))
    return theta, theta1, res


def cand_obj(
    theta: np.ndarray,
    cand_data_X: np.ndarray,
    cand_data_Y: np.ndarray,
    cand_data_T: np.ndarray,
    seldonian_type: str,
    config: SeldonianConfig,
) -> float:
    """
    Objective minimised by candidate selection: ``log_loss + penalty * max(0, u)``.

    Continuous everywhere, order 1, and equal to the log loss on the feasible side.
    See :attr:`~fair_seldonian.config.SeldonianConfig.penalty` for why a
    discontinuous barrier defeats the optimizer.
    """
    theta0 = torch.tensor(theta[:-1])
    theta1 = torch.tensor(np.array([theta[-1]]))

    log_loss = -float(f_hat(theta0, theta1, cand_data_X, cand_data_Y))
    upper_bound = float(
        ghat(
            theta0,
            theta1,
            cand_data_X,
            cand_data_Y,
            cand_data_T,
            config.candidate_ratio,
            seldonian_type,
            config,
        )
    )
    if not np.isfinite(upper_bound):
        # Degenerate split (e.g. a group vanished from this subsample): steer the
        # optimizer away rather than poisoning the search with inf/NaN.
        return float(log_loss + config.penalty)
    return float(log_loss + config.penalty * max(0.0, upper_bound))
