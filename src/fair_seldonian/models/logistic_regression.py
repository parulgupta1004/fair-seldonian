import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ..config import DEFAULT_CONFIG
from ..constraints.expression_tree import (
    construct_expr_tree_base,
    eval_expr_tree_conf_interval_base,
)
from ..constraints.expression_tree_ext import (
    construct_expr_tree,
    eval_expr_tree_conf_interval,
)


def predict(theta, theta1, X):
    """
    This is the predict function for Logistic Regression.
    Currently, it implements: 1 / (1 + e^-(X.theta + theta1))

    :param theta: The optimal theta values for the model
    :param theta1: The additional optimal theta values for the model
    :param X: The features of the dataset
    :return: The probability value of label 1 of the complete dataset
    """
    if theta1 is None or theta is None:
        return torch.ones(len(X))
    return torch.pow(
        torch.add(
            torch.exp(
                torch.mul(-1, torch.add(torch.matmul(torch.tensor(X), theta), theta1))
            ),
            1,
        ),
        -1,
    )


def f_hat(theta, theta1, X, Y):
    """
    Main objective function: negative log loss.

    :param theta: The optimal theta values for the model
    :param theta1: The additional optimal theta values for the model
    :param X: The features of the dataset
    :param Y: The true labels of the dataset
    :return: The negative log loss
    """
    pred = predict(theta, theta1, X)
    predicted_Y = torch.stack([torch.sub(1, pred), pred], dim=1)
    loss = torch.nn.CrossEntropyLoss()
    return -loss(predicted_Y, torch.tensor(Y).long())


def simple_logistic(X, Y):
    """
    Runs simple logistic regression.

    :param X: The features of the dataset
    :param Y: The true labels of the dataset
    :return: The theta values (parameters) of the model
    """
    try:
        reg = LogisticRegression(solver="lbfgs").fit(X, Y)
        theta0 = reg.intercept_[0]
        theta1 = reg.coef_[0]
        return torch.tensor(
            np.array([theta1[0], theta1[1], theta1[2], theta1[3], theta1[4]]),
            requires_grad=True,
        ), torch.tensor(np.array([theta0]), requires_grad=True)
    except Exception as e:
        print("Exception in logRes:", e)
        raise


def eval_ghat(theta, theta1, X, Y, T, seldonian_type, config=DEFAULT_CONFIG):
    if seldonian_type == "base":
        return eval_ghat_base(theta, theta1, X, Y, T, False, config)
    elif seldonian_type == "mod":
        return eval_ghat_base(theta, theta1, X, Y, T, True, config)
    elif seldonian_type == "bound":
        return eval_ghat_extend(theta, theta1, X, Y, T, True, False, False, config)
    elif seldonian_type == "const":
        return eval_ghat_extend(theta, theta1, X, Y, T, False, True, False, config)
    elif seldonian_type == "opt":
        return eval_ghat_extend(theta, theta1, X, Y, T, True, True, True, config)
    else:
        raise ValueError(f"Unknown seldonian_type: {seldonian_type}")


def ghat(
    theta, theta1, X, Y, T, candidate_ratio, seldonian_type, config=DEFAULT_CONFIG
):
    if seldonian_type == "base":
        return ghat_base(theta, theta1, X, Y, T, True, candidate_ratio, False, config)
    elif seldonian_type == "mod":
        return ghat_base(theta, theta1, X, Y, T, True, candidate_ratio, True, config)
    elif seldonian_type == "bound":
        return ghat_extend(
            theta, theta1, X, Y, T, True, candidate_ratio, True, False, False, config
        )
    elif seldonian_type == "const":
        return ghat_extend(
            theta, theta1, X, Y, T, True, candidate_ratio, False, True, False, config
        )
    elif seldonian_type == "opt":
        return ghat_extend(
            theta, theta1, X, Y, T, True, candidate_ratio, True, True, True, config
        )
    else:
        raise ValueError(f"Unknown seldonian_type: {seldonian_type}")


def ghat_base(
    theta,
    theta1,
    X,
    Y,
    T,
    predict_bound,
    candidate_ratio,
    modified_h,
    config=DEFAULT_CONFIG,
):
    pred = predict(theta, theta1, X)
    r = construct_expr_tree_base(config.constraint)
    cand_safe_ratio = None
    if candidate_ratio:
        cand_safe_ratio = (1 - candidate_ratio) / candidate_ratio
    _, u = eval_expr_tree_conf_interval_base(
        t_node=r,
        Y=Y,
        predicted_Y=pred,
        T=T,
        delta=config.delta,
        inequality=config.inequality,
        candidate_safety_ratio=cand_safe_ratio,
        predict_bound=predict_bound,
        modified_h=modified_h,
    )
    return u


def eval_ghat_base(theta, theta1, X, Y, T, modified_h, config=DEFAULT_CONFIG):
    return ghat_base(theta, theta1, X, Y, T, False, None, modified_h, config)


def ghat_extend(
    theta,
    theta1,
    X,
    Y,
    T,
    predict_bound,
    candidate_ratio,
    check_bound,
    check_const,
    modified_h,
    config=DEFAULT_CONFIG,
):
    pred = predict(theta, theta1, X)
    r = construct_expr_tree(
        config.constraint,
        config.delta,
        check_bound=check_bound,
        check_constant=check_const,
    )
    cand_safe_ratio = None
    if candidate_ratio:
        cand_safe_ratio = (1 - candidate_ratio) / candidate_ratio
    _, u = eval_expr_tree_conf_interval(
        t_node=r,
        Y=Y,
        predicted_Y=pred,
        T=T,
        inequality=config.inequality,
        candidate_safety_ratio=cand_safe_ratio,
        predict_bound=predict_bound,
        modified_h=modified_h,
    )
    return u


def eval_ghat_extend(
    theta, theta1, X, Y, T, check_bound, check_const, modified_h, config=DEFAULT_CONFIG
):
    return ghat_extend(
        theta,
        theta1,
        X,
        Y,
        T,
        False,
        None,
        check_bound,
        check_const,
        modified_h,
        config,
    )
