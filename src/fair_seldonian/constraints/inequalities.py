import math
from enum import Enum

import torch
from scipy import stats


def eval_estimate(element, Y, predicted_Y, T):
    r"""
    Estimates the value of the base variable.
    Assumes that Y and predicted_y contain 0,1 binary classification.
    Suppose we are calculating for FP(A).
    Assume X to be an indicator function defined only in case type=A
    s.t. x_i = 1 if FP occurred for ith datapoint and x_i = 0 otherwise.
    Our data samples can be assumed to be independent and identically distributed.
    Our estimate of p, \hat{p} = 1/n * \sum(x_i).
    We can safely count this as binomial random variable.
    E[\hat{p}] = 1/n * np = p.
    As we do not know p, we approximate it to \hat{p}.

    :param element: expr_tree node
    :param Y: pandas::Series
    :param predicted_Y: tensor
    :param T: pandas::Series
    :return: estimate value: float
    """
    # element will be of the form FP(A) or FN(A) or TP(A) or TN(A)
    type_attribute = element[3:-1]
    # filter predict_Y to get values where T=type_attribute and then, Y=1/0
    # average it all and return.
    type_mask = T.astype(str) == type_attribute
    Y_A = Y[type_mask]
    num_of_A = len(Y_A)
    if element.startswith("TP"):
        # filter predict_Y where Y=1
        # Predicted_y = 1 and Y=1
        label_mask = Y == 1
        mask = torch.mul(torch.tensor(type_mask), torch.tensor(label_mask))
        probs = predicted_Y[mask]
        return torch.div(torch.sum(probs), num_of_A)
    elif element.startswith("TN"):
        # filter predict_Y where Y=0
        # Predicted_y = 0 and Y=0
        label_mask = Y == 0
        mask = torch.mul(torch.tensor(type_mask), torch.tensor(label_mask))
        probs = predicted_Y[mask]
        return torch.div(torch.sum(torch.sub(1, probs)), num_of_A)
    elif element.startswith("FP"):
        # filter predict_Y where Y=0
        # Predicted_y = 1 and Y=0
        label_mask = Y == 0
        mask = torch.mul(torch.tensor(type_mask), torch.tensor(label_mask))
        probs = predicted_Y[mask]
        return torch.div(torch.sum(probs), num_of_A)
    elif element.startswith("FN"):
        # filter predict_Y where Y=1
        # Predicted_y = 0 and Y=1
        label_mask = Y == 1
        mask = torch.mul(torch.tensor(type_mask), torch.tensor(label_mask))
        probs = predicted_Y[mask]
        return torch.div(torch.sum(torch.sub(1, probs)), num_of_A)
    return None


def eval_func_bound(
    element,
    Y,
    predicted_Y,
    T,
    delta,
    inequality,
    candidate_safety_ratio,
    predict_bound,
    modified_h,
):
    estimate = eval_estimate(element, Y, predicted_Y, T)
    num_of_elements = get_num_of_elements(element, Y)
    if inequality == Inequality.T_TEST:
        variance = get_variance(element, estimate, predicted_Y, T, num_of_elements)
        if predict_bound:
            return predict_t_test(
                estimate, variance, candidate_safety_ratio * num_of_elements, delta
            )
        return eval_t_test(estimate, variance, num_of_elements, delta)
    elif inequality == Inequality.HOEFFDING_INEQUALITY:
        if predict_bound:
            if modified_h:
                return predict_hoeffding_modified(
                    estimate,
                    candidate_safety_ratio * num_of_elements,
                    num_of_elements,
                    delta,
                )
            return predict_hoeffding(
                estimate, candidate_safety_ratio * num_of_elements, delta
            )
        return eval_hoeffding(estimate, num_of_elements, delta)


####################
# Inequality class #
####################
class Inequality(Enum):
    """
    The Enum defining the inequality type.
    Currently, it supports T-test and Hoeffding.
    """

    T_TEST = 1
    HOEFFDING_INEQUALITY = 2


def get_num_of_elements(element, Y):
    if element.startswith("TP") or element.startswith("FN"):
        # filter Y=1
        return len(Y[Y == 1])
    elif element.startswith("TN") or element.startswith("FP"):
        # filter Y=0
        return len(Y[Y == 0])


def eval_hoeffding(estimate, num_of_elements, delta):
    int_size = math.sqrt(math.log(1 / delta) / (2 * num_of_elements))
    return estimate - int_size, estimate + int_size


def predict_hoeffding(estimate, safety_size, delta):
    constant_term = math.sqrt(math.log(1 / delta) / (2 * safety_size))
    int_size = 2 * constant_term
    return estimate - int_size, estimate + int_size


def predict_hoeffding_modified(estimate, num_of_elements, safety_size, delta):
    constant_term1 = math.sqrt(math.log(1 / delta) / (2 * num_of_elements))
    constant_term2 = math.sqrt(math.log(1 / delta) / (2 * safety_size))
    int_size = constant_term1 + constant_term2
    return estimate - int_size, estimate + int_size


def get_variance(element, estimate, predicted_Y, T, num_of_elements):
    # element will be of the form FP(A) or FN(A) or TP(A) or TN(A)
    type_attribute = element[3:-1]
    type_Y = predicted_Y[T.astype(str) == type_attribute]
    sum_term = (type_Y - estimate) ** 2
    return math.sqrt(float(sum_term.sum()) / (num_of_elements - 1))


def eval_t_test(estimate, variance, num_of_elements, delta):
    t = stats.t.ppf(1 - delta, num_of_elements - 1)
    int_size = (variance / math.sqrt(num_of_elements)) * t
    return estimate - int_size, estimate + int_size


def predict_t_test(estimate, variance, safety_size, delta):
    t = stats.t.ppf(1 - delta, safety_size - 1)
    int_size = 2 * (variance / math.sqrt(safety_size)) * t
    return estimate - int_size, estimate + int_size
