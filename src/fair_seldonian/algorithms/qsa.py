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
    This function is used to run the qsa (Quasi-Seldonian Algorithm)

    :param X: The features of the dataset
    :param Y: The corresponding labels of the dataset
    :param T: The corresponding sensitive attributes of the dataset
    :param seldonian_type: The mode used in the experiment
    :param init_sol: The initial theta values for the model
    :param init_sol1: The additional initial theta values for the model
    :param config: Algorithm configuration
    :return: (thetexit
    a, theta1, passed_safety) tuple
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
    :return: Bool value of whether the candidate solution passed safety test or not.
    """
    upper_bound = eval_ghat(
        theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type, config
    )
    logger.debug(f"Safety test upperbound: {upper_bound}")
    if upper_bound > 0.0:
        return False
    return True


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
    if init_sol is None:
        init_sol, init_sol1 = simple_logistic(cand_data_X, cand_data_Y)
    # init_sol and init_sol1 are always supplied (or recomputed) as a pair.
    assert init_sol1 is not None
    if logger.isEnabledFor(logging.DEBUG):
        init_upper_bound = eval_ghat(
            init_sol,
            init_sol1,
            cand_data_X,
            cand_data_Y,
            cand_data_T,
            seldonian_type,
            config,
        )
        logger.debug(f"Initial LS upperbound: {init_upper_bound}")
    theta = init_sol.detach().numpy()
    theta1 = init_sol1.detach().numpy()
    init_theta = np.concatenate((theta, theta1))
    res = minimize(
        cand_obj,
        x0=init_theta,
        method="Powell",
        options={"disp": False, "maxiter": 10000},
        args=(cand_data_X, cand_data_Y, cand_data_T, seldonian_type, config),
    )
    theta_numpy = res.x[:-1]
    theta1_numpy = res.x[-1]
    theta0 = torch.tensor(theta_numpy)
    theta1 = torch.tensor(np.array([theta1_numpy]))
    return theta0, theta1


def cand_obj(
    theta: np.ndarray,
    cand_data_X: np.ndarray,
    cand_data_Y: np.ndarray,
    cand_data_T: np.ndarray,
    seldonian_type: str,
    config: SeldonianConfig,
) -> float:
    """
    Objective function minimized by the optimizer.

    :param theta: The theta values for the model
    :param cand_data_X: The features of the candidate dataset
    :param cand_data_Y: The corresponding labels of the candidate dataset
    :param cand_data_T: The corresponding sensitive attributes of the candidate dataset
    :param seldonian_type: The mode used in the experiment
    :param config: Algorithm configuration
    :return: The objective value.
    """
    theta_numpy = theta[:-1]
    theta1_numpy = theta[-1]
    theta0 = torch.tensor(theta_numpy)
    theta1 = torch.tensor(np.array([theta1_numpy]))

    result = f_hat(theta0, theta1, cand_data_X, cand_data_Y)
    upper_bound = ghat(
        theta0,
        theta1,
        cand_data_X,
        cand_data_Y,
        cand_data_T,
        config.candidate_ratio,
        seldonian_type,
        config,
    )

    if upper_bound > 0.0:
        result = -10000.0 - upper_bound
    return float(-result)
