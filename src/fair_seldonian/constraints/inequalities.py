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


def parse_base_token(element: str) -> tuple[str, str]:
    """Split ``"TP(A)"`` into ``("TP", "A")``."""
    head, _, rest = element.partition("(")
    return head, rest[:-1]


def contributions(
    element: str, Y: Array, predicted_Y: torch.Tensor, T: Array
) -> torch.Tensor:
    r"""Per-sample contributions to the base variable ``element``, over its group.

    ``TP(A)``, ``FP(A)``, ``TN(A)``, ``FN(A)`` are *fractions of group* ``A``: the
    four cells sum to 1 within a group. So ``TP(A)`` estimates
    :math:`P(\hat{Y}=1, Y=1 \mid T=A)` - a joint probability, **not** the
    true-positive rate :math:`P(\hat{Y}=1 \mid Y=1, T=A)`. Conditional rates are
    built from these cells by division; see
    :mod:`fair_seldonian.constraints.fairness`.

    Concretely, for ``TP(A)`` this returns the vector

    .. math:: x_i = \mathbb{1}[Y_i = 1] \cdot \hat{p}_i, \quad i \in \{T = A\}

    whose mean is the estimate and whose length is the sample size that any
    concentration inequality must use. Returning the raw vector (rather than the
    mean alone) is deliberate: the estimate, its sample size and its variance are
    then guaranteed to refer to the same index set. Computing them separately
    invites a mismatch that is hard to see and unsound: building the interval from
    ``#{Y == 1}`` over *all* groups while the estimate divides by the size of one
    group makes the interval anti-conservative whenever that group is small
    relative to the dataset.

    Conditioning on the group-assignment vector ``T``, these :math:`|A|` terms are
    i.i.d. and lie in :math:`[0, 1]`, which is what Hoeffding requires.

    :param element: base-variable token, e.g. ``"TP(A)"``.
    :param Y: true labels.
    :param predicted_Y: predicted probability of label 1, for the whole dataset.
    :param T: sensitive attribute column.
    :return: 1-D tensor of per-sample contributions, of length ``#{T == group}``.
    """
    measure, group = parse_base_token(element)
    in_group = torch.tensor(group_mask(T, group))
    probs = predicted_Y[in_group]
    labels = torch.tensor(Y)[in_group]

    if measure == "TP":  # predicted 1, actually 1
        return probs * (labels == 1)
    if measure == "FP":  # predicted 1, actually 0
        return probs * (labels == 0)
    if measure == "TN":  # predicted 0, actually 0
        return (1 - probs) * (labels == 0)
    if measure == "FN":  # predicted 0, actually 1
        return (1 - probs) * (labels == 1)
    raise ValueError(f"Unknown constraint variable: {element!r}")


def eval_estimate(
    element: str, Y: Array, predicted_Y: torch.Tensor, T: Array
) -> torch.Tensor:
    """Point estimate of the base variable ``element``.

    This is the mean of :func:`contributions`. See that function for what the
    quantity means and why the two are tied together.

    :param element: base-variable token, e.g. ``"TP(A)"``.
    :param Y: true labels.
    :param predicted_Y: predicted probability of label 1.
    :param T: sensitive attribute column.
    :return: scalar tensor.
    """
    x = contributions(element, Y, predicted_Y, T)
    if x.numel() == 0:
        # No samples in this group: the fraction is undefined. Return 0 rather than
        # dividing by zero (which would yield NaN and silently corrupt downstream
        # arithmetic). The confidence-bound path guards this case separately and
        # fails closed; see eval_func_bound.
        return torch.tensor(0.0)
    return x.mean()


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
    """Confidence interval for a single base variable.

    :param delta: failure probability budget allocated to this node.
    """
    x = contributions(element, Y, predicted_Y, T)
    n = int(x.numel())
    # When the group is empty, or too small to form an interval, the estimate and
    # its confidence bound are undefined. Return the widest possible interval so the
    # constraint's upper bound becomes +inf and the safety test fails closed, instead
    # of dividing by zero or silently propagating NaN and wrongly passing safety.
    min_required = 1 if inequality == Inequality.HOEFFDING_INEQUALITY else 2
    if n < min_required:
        return -math.inf, math.inf

    estimate = x.mean()

    if inequality == Inequality.HOEFFDING_INEQUALITY:
        if predict_bound:
            # predict_bound is only set together with a candidate/safety split.
            assert candidate_safety_ratio is not None
            safety_size = candidate_safety_ratio * n
            if modified_h:
                return predict_hoeffding_modified(estimate, safety_size, n, delta)
            return predict_hoeffding(estimate, safety_size, delta)
        return eval_hoeffding(estimate, n, delta)

    if inequality == Inequality.T_TEST:
        std = float(x.detach().std(unbiased=True))
        if predict_bound:
            assert candidate_safety_ratio is not None
            return predict_t_test(estimate, std, candidate_safety_ratio * n, delta)
        return eval_t_test(estimate, std, n, delta)

    raise ValueError(f"Unknown inequality: {inequality!r}")


####################
# Inequality class #
####################
class Inequality(Enum):
    """The concentration inequality used to build confidence intervals.

    ``HOEFFDING_INEQUALITY`` is distribution-free and gives a genuine
    high-confidence guarantee. ``T_TEST`` assumes approximate normality of the
    sample mean, which makes the result *quasi*-Seldonian: the guarantee is only
    as good as that approximation.
    """

    T_TEST = 1
    HOEFFDING_INEQUALITY = 2


#############
# Hoeffding #
#############
def eval_hoeffding(
    estimate: Bound, num_of_elements: float, delta: float
) -> tuple[Bound, Bound]:
    """Hoeffding interval for a mean of ``num_of_elements`` i.i.d. terms in [0, 1]."""
    int_size = math.sqrt(math.log(1 / delta) / (2 * num_of_elements))
    return estimate - int_size, estimate + int_size


def predict_hoeffding(
    estimate: Bound, safety_size: float, delta: float
) -> tuple[Bound, Bound]:
    """Candidate-selection *prediction* of the safety-test interval.

    This is a heuristic, not a bound: it guesses whether the safety test will pass
    so that candidate selection can avoid proposing solutions that would be
    rejected. It carries no guarantee and needs none - the safety test supplies the
    guarantee on its own.

    Following Thomas et al. (2019), the safety-set interval is inflated by 2 to
    stay conservative.
    """
    int_size = 2 * math.sqrt(math.log(1 / delta) / (2 * safety_size))
    return estimate - int_size, estimate + int_size


def predict_hoeffding_modified(
    estimate: Bound, num_of_elements: float, safety_size: float, delta: float
) -> tuple[Bound, Bound]:
    """Split the prediction into candidate error and safety-set width.

    Replaces the blanket factor of 2 in :func:`predict_hoeffding` with the sum of a
    term for the candidate-set estimation error and a term for the safety-set
    interval. Like :func:`predict_hoeffding` this is a heuristic and affects only
    which candidates are proposed.
    """
    constant_term1 = math.sqrt(math.log(1 / delta) / (2 * num_of_elements))
    constant_term2 = math.sqrt(math.log(1 / delta) / (2 * safety_size))
    int_size = constant_term1 + constant_term2
    return estimate - int_size, estimate + int_size


##########
# t-test #
##########
def eval_t_test(
    estimate: Bound, std: float, num_of_elements: float, delta: float
) -> tuple[Bound, Bound]:
    """Student's t interval. ``std`` is the unbiased sample standard deviation."""
    t = float(stats.t.ppf(1 - delta, num_of_elements - 1))
    int_size = (std / math.sqrt(num_of_elements)) * t
    return estimate - int_size, estimate + int_size


def predict_t_test(
    estimate: Bound, std: float, safety_size: float, delta: float
) -> tuple[Bound, Bound]:
    """Candidate-selection prediction of the t-test safety interval."""
    t = float(stats.t.ppf(1 - delta, safety_size - 1))
    int_size = 2 * (std / math.sqrt(safety_size)) * t
    return estimate - int_size, estimate + int_size
