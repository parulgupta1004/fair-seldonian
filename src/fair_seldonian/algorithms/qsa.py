from __future__ import annotations

import logging
from typing import cast

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from ..config import DEFAULT_CONFIG, SeldonianConfig
from ..models.logistic_regression import eval_ghat, f_hat, ghat, simple_logistic

logger = logging.getLogger(__name__)


def QSA(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    seldonian_type: str,
    init_sol: torch.Tensor | None,
    init_sol1: torch.Tensor | None,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
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
    :return: (theta, theta1, passed_safety) tuple
    """
    cand_data_X, safe_data_X, cand_data_Y, safe_data_Y = cast(
        "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
        train_test_split(X, Y, test_size=1 - config.candidate_ratio, shuffle=False),
    )
    cand_data_T, safe_data_T = np.split(
        T,
        [
            int(config.candidate_ratio * T.size),
        ],
    )

    theta, theta1 = get_cand_solution(
        cand_data_X,
        cand_data_Y,
        cand_data_T,
        seldonian_type,
        init_sol,
        init_sol1,
        config,
    )

    if logger.isEnabledFor(logging.DEBUG):
        cand_upper_bound = eval_ghat(
            theta, theta1, cand_data_X, cand_data_Y, cand_data_T, seldonian_type, config
        )
        logger.debug(f"Actual cand sol upperbound: {cand_upper_bound}")
    passed_safety = safety_test(
        theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type, config
    )
    return theta, theta1, passed_safety


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
    return bool(upper_bound <= 0.0)


def get_cand_solution(
    cand_data_X: np.ndarray,
    cand_data_Y: np.ndarray,
    cand_data_T: np.ndarray,
    seldonian_type: str,
    init_sol: torch.Tensor | None,
    init_sol1: torch.Tensor | None,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function provides the candidate solution.

    :param cand_data_X: The features of the candidate dataset
    :param cand_data_Y: The corresponding labels of the candidate dataset
    :param cand_data_T: The corresponding sensitive attributes of the candidate dataset
    :param seldonian_type: The mode used in the experiment
    :param init_sol: The initial theta values for the model
    :param init_sol1: The additional initial theta values for the model
    :param config: Algorithm configuration
    :return: The candidate solution (theta, theta1).
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
    return theta, theta1


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
