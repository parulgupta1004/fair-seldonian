import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from ..config import DEFAULT_CONFIG
from ..models.logistic_regression import eval_ghat, fHat, ghat, simple_logistic


def QSA(X, Y, T, seldonian_type, init_sol, init_sol1, config=DEFAULT_CONFIG):
    """
    This function is used to run the qsa (Quasi-Seldonian Algorithm)

    :param X: The features of the dataset
    :param Y: The corresponding labels of the dataset
    :param T: The corresponding sensitive attributes of the dataset
    :param seldonian_type: The mode used in the experiment
    :param init_sol: The initial theta values for the model
    :param init_sol1: The additional initial theta values for the model
    :param config: Algorithm configuration
    :return: (theta, theta1, passed_safety) tuple
    """
    cand_data_X, safe_data_X, cand_data_Y, safe_data_Y = train_test_split(
        X, Y, test_size=1 - config.candidate_ratio, shuffle=False
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
    print(
        "Actual cand sol upperbound: ",
        eval_ghat(
            theta, theta1, cand_data_X, cand_data_Y, cand_data_T, seldonian_type, config
        ),
    )
    passed_safety = safety_test(
        theta, theta1, safe_data_X, safe_data_Y, safe_data_T, seldonian_type, config
    )
    return [theta, theta1, passed_safety]


def safety_test(
    theta,
    theta1,
    safe_data_X,
    safe_data_Y,
    safe_data_T,
    seldonian_type,
    config=DEFAULT_CONFIG,
):
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
    print("Safety test upperbound: ", upper_bound)
    if upper_bound > 0.0:
        return False
    return True


def get_cand_solution(
    cand_data_X,
    cand_data_Y,
    cand_data_T,
    seldonian_type,
    init_sol,
    init_sol1,
    config=DEFAULT_CONFIG,
):
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
    print(
        "Initial LS upperbound: ",
        eval_ghat(
            init_sol,
            init_sol1,
            cand_data_X,
            cand_data_Y,
            cand_data_T,
            seldonian_type,
            config,
        ),
    )
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


def cand_obj(theta, cand_data_X, cand_data_Y, cand_data_T, seldonian_type, config):
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

    result = fHat(theta0, theta1, cand_data_X, cand_data_Y)
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


def _get_cand_solution2(
    cand_data_X, cand_data_Y, cand_data_T, seldonian_type, config=DEFAULT_CONFIG
):
    init_sol, init_sol1 = simple_logistic(cand_data_X, cand_data_Y)
    init_fhat = fHat(init_sol, init_sol1, cand_data_X, cand_data_Y)
    init_ghat = eval_ghat(
        init_sol,
        init_sol1,
        cand_data_X,
        cand_data_Y,
        cand_data_T,
        seldonian_type,
        config,
    )
    init_fhat.backward()
    assert init_sol.grad is not None and init_sol1.grad is not None
    numerator = init_sol.grad + init_sol1.grad
    init_ghat.backward()
    assert init_sol.grad is not None and init_sol1.grad is not None
    denominator = init_sol.grad + init_sol1.grad
    lambda_value = -numerator / denominator
    fin_lambda = None
    for i in range(len(init_sol + 1)):
        if lambda_value[i] > 0:
            fin_lambda = float(lambda_value[i])
            break
    if not fin_lambda:
        fin_lambda = 1
    print(
        "Initial LS upperbound: ",
        eval_ghat(
            init_sol,
            init_sol1,
            cand_data_X,
            cand_data_Y,
            cand_data_T,
            seldonian_type,
            config,
        ),
    )
    theta = init_sol.detach().numpy()
    theta1 = init_sol1.detach().numpy()
    init_theta = np.concatenate((theta, theta1))
    res = minimize(
        _cand_obj2,
        x0=init_theta,
        method="BFGS",
        options={"disp": False, "maxiter": 12000},
        args=(
            cand_data_X,
            cand_data_Y,
            cand_data_T,
            seldonian_type,
            fin_lambda,
            config,
        ),
    )
    theta_numpy = res.x[:-1]
    theta1_numpy = res.x[-1]
    theta0 = torch.tensor(theta_numpy)
    theta1 = torch.tensor(np.array([theta1_numpy]))
    return theta0, theta1


def _cand_obj2(
    theta,
    cand_data_X,
    cand_data_Y,
    cand_data_T,
    seldonian_type,
    lambda_value,
    config,
):
    theta_numpy = theta[:-1]
    theta1_numpy = theta[-1]
    theta0 = torch.tensor(theta_numpy)
    theta1 = torch.tensor(np.array([theta1_numpy]))

    result = fHat(theta0, theta1, cand_data_X, cand_data_Y)
    upper_bound = eval_ghat(
        theta0, theta1, cand_data_X, cand_data_Y, cand_data_T, seldonian_type, config
    )
    if upper_bound > 0:
        result = float(-1000 - (lambda_value * upper_bound))
    else:
        result = float(-result - (lambda_value * upper_bound))
    return float(-result)
