from __future__ import annotations

import math
import threading
from enum import Enum
from typing import TYPE_CHECKING

import torch
from scipy import stats

if TYPE_CHECKING:
    from .._typing import Array, Bound

# `T.astype(str) == group` dominates the confidence-bound hot path: the optimizer
# evaluates the constraint thousands of times while T never changes. Memoize the
# group mask per (T, group). Keyed by object identity and guarded with `is`, so a
# recycled id can never return a stale mask; T is treated as immutable here. The
# cache is bounded and cleared wholesale to cap retained references.
_GROUP_MASK_CACHE: dict[tuple[int, str], tuple[Array, Array]] = {}
_GROUP_MASK_CACHE_MAX = 32
_GROUP_MASK_CACHE_LOCK = threading.Lock()  # to ensure it works on free-threading Python


def group_mask(T: Array, group: str) -> Array:
    """Boolean mask of rows whose sensitive attribute equals ``group`` (cached)."""
    key = (id(T), group)
    cached = _GROUP_MASK_CACHE.get(key)
    if cached is not None and cached[0] is T:
        return cached[1]
    mask = T.astype(str) == group
    with _GROUP_MASK_CACHE_LOCK:
        if len(_GROUP_MASK_CACHE) >= _GROUP_MASK_CACHE_MAX:
            _GROUP_MASK_CACHE.clear()
        _GROUP_MASK_CACHE[key] = (T, mask)
    return mask


def eval_estimate(
    element: str, Y: Array, predicted_Y: torch.Tensor, T: Array
) -> torch.Tensor:
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
    type_mask = group_mask(T, type_attribute)
    Y_A = Y[type_mask]
    num_of_A = len(Y_A)
    if num_of_A == 0:
        # No samples in this group: the rate is undefined. Return 0 rather than
        # dividing by zero (which would yield NaN and silently corrupt downstream
        # arithmetic). The confidence-bound path guards this case separately and
        # fails closed; see eval_func_bound.
        return torch.tensor(0.0)
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
    raise ValueError(f"Unknown constraint variable: {element!r}")


def eval_func_bound(
    element: str,
    Y: Array,
    predicted_Y: torch.Tensor,
    T: Array,
    delta: float,
    inequality: Inequality,
    candidate_safety_ratio: float | None,
    predict_bound: bool,
    modified_h: bool,
) -> tuple[Bound, Bound]:
    num_of_elements = get_num_of_elements(element, Y)
    num_in_group = int(group_mask(T, element[3:-1]).sum())
    # When the group is empty or there are too few label-matched samples to form
    # an interval, the estimate and its confidence bound are undefined. Return the
    # widest possible interval so the constraint's upper bound becomes +inf and the
    # safety test fails closed, instead of dividing by zero (Hoeffding/variance) or
    # silently propagating NaN through the bound and wrongly passing safety.
    min_required = 2 if inequality == Inequality.T_TEST else 1
    if num_in_group == 0 or num_of_elements < min_required:
        return -math.inf, math.inf
    estimate = eval_estimate(element, Y, predicted_Y, T)
    if inequality == Inequality.T_TEST:
        variance = get_variance(element, estimate, predicted_Y, T, num_of_elements)
        if predict_bound:
            # predict_bound is only set together with a candidate/safety split.
            assert candidate_safety_ratio is not None
            return predict_t_test(
                estimate, variance, candidate_safety_ratio * num_of_elements, delta
            )
        return eval_t_test(estimate, variance, num_of_elements, delta)
    elif inequality == Inequality.HOEFFDING_INEQUALITY:
        if predict_bound:
            # predict_bound is only set together with a candidate/safety split.
            assert candidate_safety_ratio is not None
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
    raise ValueError(f"Unknown inequality: {inequality!r}")


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


def get_num_of_elements(element: str, Y: Array) -> int:
    if element.startswith("TP") or element.startswith("FN"):
        # filter Y=1
        return len(Y[Y == 1])
    elif element.startswith("TN") or element.startswith("FP"):
        # filter Y=0
        return len(Y[Y == 0])
    raise ValueError(f"Unknown constraint variable: {element!r}")


def eval_hoeffding(
    estimate: Bound, num_of_elements: int, delta: float
) -> tuple[Bound, Bound]:
    int_size = math.sqrt(math.log(1 / delta) / (2 * num_of_elements))
    return estimate - int_size, estimate + int_size


def predict_hoeffding(
    estimate: Bound, safety_size: float, delta: float
) -> tuple[Bound, Bound]:
    constant_term = math.sqrt(math.log(1 / delta) / (2 * safety_size))
    int_size = 2 * constant_term
    return estimate - int_size, estimate + int_size


def predict_hoeffding_modified(
    estimate: Bound, num_of_elements: float, safety_size: float, delta: float
) -> tuple[Bound, Bound]:
    constant_term1 = math.sqrt(math.log(1 / delta) / (2 * num_of_elements))
    constant_term2 = math.sqrt(math.log(1 / delta) / (2 * safety_size))
    int_size = constant_term1 + constant_term2
    return estimate - int_size, estimate + int_size


def get_variance(
    element: str,
    estimate: Bound,
    predicted_Y: torch.Tensor,
    T: Array,
    num_of_elements: int,
) -> float:
    # element will be of the form FP(A) or FN(A) or TP(A) or TN(A)
    type_attribute = element[3:-1]
    type_Y = predicted_Y[torch.tensor(group_mask(T, type_attribute))]
    sum_term = (type_Y - estimate) ** 2
    return math.sqrt(float(sum_term.sum().detach()) / (num_of_elements - 1))


def eval_t_test(
    estimate: Bound, variance: float, num_of_elements: int, delta: float
) -> tuple[Bound, Bound]:
    t = float(stats.t.ppf(1 - delta, num_of_elements - 1))
    int_size = (variance / math.sqrt(num_of_elements)) * t
    return estimate - int_size, estimate + int_size


def predict_t_test(
    estimate: Bound, variance: float, safety_size: float, delta: float
) -> tuple[Bound, Bound]:
    t = float(stats.t.ppf(1 - delta, safety_size - 1))
    int_size = 2 * (variance / math.sqrt(safety_size)) * t
    return estimate - int_size, estimate + int_size
