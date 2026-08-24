from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
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

logger = logging.getLogger(__name__)


def check_constraint_groups(groups: list[str], T: Array) -> None:
    """Raise if a group named in the constraint matches no row of ``T``.

    Group labels are compared as strings, so an integer column and a float column
    behave differently: ``str(1) == "1"`` but ``str(1.0) == "1.0"``. Passing ``T``
    through anything that upcasts to float - ``DataFrame.values`` on a frame with
    any float column, for instance - therefore makes every mask empty. The bound
    then fails closed to ``+inf``, the safety test rejects everything, and the run
    looks like a legitimate "no solution found" rather than a type error. Call this
    once up front so the mistake is loud.
    """
    present = {str(v) for v in np.unique(np.asarray(T))}
    missing = sorted(set(groups) - present)
    if missing:
        raise ValueError(
            f"constraint refers to group(s) {missing} that appear in no row of T; "
            f"T contains {sorted(present)[:10]}. Group labels are matched as "
            "strings, so check that T has not been upcast (e.g. int 1 -> float 1.0)."
        )


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
    """Split ``"TPR(F)"`` into ``("TPR", "F")``."""
    head, _, rest = element.partition("(")
    return head, rest[:-1]


def conditioning_set(element: str) -> tuple[str, int | None]:
    r"""Which rows a base variable is a mean over.

    A cell such as ``TP(g)`` is a fraction of the whole of group ``g``: every row
    of the group contributes, rows with the wrong label contributing zero. A rate
    such as ``TPR(g)`` is a mean over only the rows of group ``g`` with
    :math:`Y = 1`. The distinction sets the sample size an inequality may use, and
    it decides which base variables are independent of one another: two variables
    are independent exactly when their conditioning sets are disjoint.
    """
    measure, group = parse_base_token(element)
    if measure in ("TPR", "FNR"):
        return group, 1
    if measure in ("FPR", "TNR"):
        return group, 0
    return group, None


def contributions(
    element: str, Y: Array, predicted_Y: torch.Tensor, T: Array
) -> torch.Tensor:
    r"""Per-sample contributions to the base variable ``element``, over its group.

    ``TP(A)``, ``FP(A)``, ``TN(A)``, ``FN(A)`` are *fractions of group* ``A``: the
    four cells sum to 1 within a group. So ``TP(A)`` estimates
    :math:`P(\hat{Y}=1, Y=1 \mid T=A)` - a joint probability, **not** the
    true-positive rate :math:`P(\hat{Y}=1 \mid Y=1, T=A)`.

    ``PR(A)`` and ``NR(A)`` are the predicted-positive and predicted-negative
    rates, ``TP + FP`` and ``TN + FN``. They ignore the label, so they are means
    over the whole group too and likewise sum to 1.

    ``TPR``, ``FPR``, ``TNR`` and ``FNR`` are the label-conditioned rates, means
    over a *subset* of the group; see :func:`conditioning_set`.

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
    i.i.d. and lie in :math:`[0, 1]`, which is what Hoeffding and empirical
    Bernstein require.

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
    # Predicted-positive and predicted-negative rate. Unlike the cells these
    # ignore the label entirely: PR(g) is TP(g) + FP(g), so it is still a mean
    # over the whole of group g and shares the cells' conditioning set.
    if measure == "PR":  # predicted 1, label irrelevant
        return probs
    if measure == "NR":  # predicted 0, label irrelevant
        return 1 - probs
    # Rates condition on the label, so they are means over a subset of the group
    # and carry their own, smaller sample size.
    if measure == "TPR":
        return probs[labels == 1]
    if measure == "FNR":
        return (1 - probs)[labels == 1]
    if measure == "FPR":
        return probs[labels == 0]
    if measure == "TNR":
        return (1 - probs)[labels == 0]
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
    two_sided: bool = True,
) -> tuple[Bound, Bound]:
    """Confidence interval for a single base variable.

    :param delta: failure probability budget allocated to this node.
    :param two_sided: whether the caller consumes *both* endpoints. A symmetric
        interval that must cover on both sides needs ``ln(2/delta)``; one that is
        only ever read from above needs ``ln(1/delta)``. Passing ``False`` when the
        lower endpoint is in fact used would silently double the true failure
        probability, so the sidedness is derived structurally from the constraint
        tree rather than guessed - see
        :func:`fair_seldonian.constraints.expression_tree.child_sides`.
    """
    x = contributions(element, Y, predicted_Y, T)
    n = int(x.numel())
    # When the group is empty, or too small to form an interval, the estimate and
    # its confidence bound are undefined. Return the widest possible interval so the
    # constraint's upper bound becomes +inf and the safety test fails closed, instead
    # of dividing by zero or silently propagating NaN and wrongly passing safety.
    min_required = (
        1
        if inequality
        in (
            Inequality.HOEFFDING_INEQUALITY,
            Inequality.BETTING,
        )
        else 2
    )
    if n < min_required:
        if n == 0:
            logger.warning(
                "group %r of %r matched no rows; the bound will be +inf and the "
                "safety test will reject. If this is unexpected, check the dtype "
                "of T (see check_constraint_groups).",
                parse_base_token(element)[1],
                element,
            )
        return -math.inf, math.inf

    estimate = x.mean()

    if inequality == Inequality.HOEFFDING_INEQUALITY:
        if predict_bound:
            # predict_bound is only set together with a candidate/safety split.
            assert candidate_safety_ratio is not None
            safety_size = candidate_safety_ratio * n
            if modified_h:
                return predict_hoeffding_modified(
                    estimate, safety_size, n, delta, two_sided
                )
            return predict_hoeffding(estimate, safety_size, delta, two_sided)
        return eval_hoeffding(estimate, n, delta, two_sided)

    if inequality == Inequality.T_TEST:
        std = float(x.detach().std(unbiased=True))
        if predict_bound:
            assert candidate_safety_ratio is not None
            return predict_t_test(estimate, std, candidate_safety_ratio * n, delta)
        return eval_t_test(estimate, std, n, delta, two_sided)

    if inequality == Inequality.BETTING:
        values = x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)
        lo, hi = betting_interval(values, delta, two_sided)
        if predict_bound:
            # Candidate selection predicts the safety-test interval; widen by the
            # usual factor of 2 about the estimate, as for the other inequalities.
            centre = float(estimate)
            return centre - 2 * (centre - lo), centre + 2 * (hi - centre)
        return lo, hi

    if inequality == Inequality.EMPIRICAL_BERNSTEIN:
        var = float(x.detach().var(unbiased=True))
        if predict_bound:
            assert candidate_safety_ratio is not None
            return predict_empirical_bernstein(
                estimate, var, candidate_safety_ratio * n, delta, two_sided
            )
        return eval_empirical_bernstein(estimate, var, n, delta, two_sided)

    raise ValueError(f"Unknown inequality: {inequality!r}")


####################
# Inequality class #
####################
class Inequality(Enum):
    """The concentration inequality used to build confidence intervals.

    ``HOEFFDING_INEQUALITY`` and ``EMPIRICAL_BERNSTEIN`` are distribution-free and
    give a genuine high-confidence guarantee. ``T_TEST`` assumes approximate
    normality of the sample mean, which makes the result *quasi*-Seldonian: the
    guarantee is only as good as that approximation.

    Empirical Bernstein pays for the variance it measures rather than assuming the
    worst case of 1/4, so it is much tighter for base variables whose value is far
    from 1/2 and slightly *looser* at exactly 1/2. Pick it when a group's rate is
    small - which is the common case for a minority group.

    ``BETTING`` is the Waudby-Smith and Ramdas betting interval, which is also
    distribution-free and dominates the other two for bounded variables: it adapts
    to the observed variance like empirical Bernstein but without paying the
    additive penalty term. Locating each endpoint takes repeated passes over the
    data, so it is the slowest of the four.
    """

    T_TEST = 1
    HOEFFDING_INEQUALITY = 2
    EMPIRICAL_BERNSTEIN = 3
    BETTING = 4


@dataclass(frozen=True)
class InequalityInfo:
    """What an :class:`Inequality` assumes and what it therefore guarantees.

    Held beside the enum rather than written into the documentation, because the
    documentation's comparison table is generated from it: adding a member
    without describing it here fails ``tests/test_docs_sync.py``.
    """

    assumption: str
    distribution_free: bool
    summary: str


#: Per-member metadata, keyed by :class:`Inequality`. The docs render this table.
INEQUALITY_INFO: dict[Inequality, InequalityInfo] = {
    Inequality.HOEFFDING_INEQUALITY: InequalityInfo(
        assumption="each term lies in [0, 1]",
        distribution_free=True,
        summary=(
            "The default. Assumes the worst possible variance, 1/4, so it is exactly "
            "right at a base rate of 1/2 and increasingly pessimistic away from it."
        ),
    ),
    Inequality.EMPIRICAL_BERNSTEIN: InequalityInfo(
        assumption="each term lies in [0, 1]",
        distribution_free=True,
        summary=(
            "Pays for the variance it measures rather than the worst case, plus an "
            "additive term for having estimated that variance from the same sample. "
            "Tighter than Hoeffding whenever the sample variance is below 1/4, and "
            "slightly looser at exactly 1/2."
        ),
    ),
    Inequality.BETTING: InequalityInfo(
        assumption="each term lies in [0, 1]",
        distribution_free=True,
        summary=(
            "Inverts a betting martingale test via Ville's inequality. Adapts to the "
            "observed variance like empirical Bernstein without paying the additive "
            "penalty, so it dominates the other two for bounded variables. Locating "
            "each endpoint takes repeated passes over the data."
        ),
    ),
    Inequality.T_TEST: InequalityInfo(
        assumption="the sample mean is approximately normal",
        distribution_free=False,
        summary=(
            "An appeal to the central limit theorem rather than a finite-sample "
            "bound, which is what makes the result *quasi*-Seldonian. Included "
            "because the reference work uses it; prefer one of the three above for "
            "any claim that has to hold."
        ),
    ),
}


def _log_term(delta: float, two_sided: bool) -> float:
    """``ln(2/delta)`` for a two-sided interval, ``ln(1/delta)`` for one-sided.

    A symmetric ``estimate +/- w`` interval fails if *either* side is breached, so
    the budget must be split between them. Using ``ln(1/delta)`` for a two-sided
    interval delivers coverage ``1 - 2*delta``, not ``1 - delta``.
    """
    return math.log((2.0 if two_sided else 1.0) / delta)


#############
# Hoeffding #
#############
def eval_hoeffding(
    estimate: Bound, num_of_elements: float, delta: float, two_sided: bool = True
) -> tuple[Bound, Bound]:
    """Hoeffding interval for a mean of ``num_of_elements`` i.i.d. terms in [0, 1]."""
    int_size = math.sqrt(_log_term(delta, two_sided) / (2 * num_of_elements))
    return estimate - int_size, estimate + int_size


def predict_hoeffding(
    estimate: Bound, safety_size: float, delta: float, two_sided: bool = True
) -> tuple[Bound, Bound]:
    """Candidate-selection *prediction* of the safety-test interval.

    This is a heuristic, not a bound: it guesses whether the safety test will pass
    so that candidate selection can avoid proposing solutions that would be
    rejected. It carries no guarantee and needs none - the safety test supplies the
    guarantee on its own. (It could not be a bound in any case: ``theta_c`` is
    chosen by optimizing on the candidate data, so the candidate-set estimate is
    not unbiased and pointwise concentration does not apply.)

    Following Thomas et al. (2019), the safety-set interval is inflated by 2 to
    stay conservative.
    """
    int_size = 2 * math.sqrt(_log_term(delta, two_sided) / (2 * safety_size))
    return estimate - int_size, estimate + int_size


def predict_hoeffding_modified(
    estimate: Bound,
    safety_size: float,
    candidate_size: float,
    delta: float,
    two_sided: bool = True,
) -> tuple[Bound, Bound]:
    """Split the prediction into candidate error and safety-set width.

    Replaces the blanket factor of 2 in :func:`predict_hoeffding` with the sum of a
    term for the candidate-set estimation error and a term for the safety-set
    interval. With a 60:40 candidate:safety split the factor becomes
    ``1 + sqrt(0.4/0.6) = 1.82`` rather than 2, so it is a mild relaxation - and it
    matters more the more lopsided the split is. Like :func:`predict_hoeffding`
    this is a heuristic and affects only which candidates are proposed.
    """
    log_term = _log_term(delta, two_sided)
    int_size = math.sqrt(log_term / (2 * safety_size)) + math.sqrt(
        log_term / (2 * candidate_size)
    )
    return estimate - int_size, estimate + int_size


#######################
# Empirical Bernstein #
#######################
def eval_empirical_bernstein(
    estimate: Bound,
    variance: float,
    num_of_elements: float,
    delta: float,
    two_sided: bool = True,
) -> tuple[Bound, Bound]:
    """Maurer-Pontil empirical Bernstein interval for terms in [0, 1].

    ``variance`` is the unbiased sample variance. Tighter than Hoeffding whenever
    the sample variance is below the worst case of 1/4.
    """
    if num_of_elements < 2:
        return -math.inf, math.inf
    log_term = _log_term(delta, two_sided)
    int_size = math.sqrt(2 * variance * log_term / num_of_elements) + (
        7 * log_term / (3 * (num_of_elements - 1))
    )
    return estimate - int_size, estimate + int_size


def predict_empirical_bernstein(
    estimate: Bound,
    variance: float,
    safety_size: float,
    delta: float,
    two_sided: bool = True,
) -> tuple[Bound, Bound]:
    """Candidate-selection prediction of the empirical-Bernstein safety interval."""
    lo, hi = eval_empirical_bernstein(estimate, variance, safety_size, delta, two_sided)
    int_size = 2 * (hi - estimate)
    return estimate - int_size, estimate + int_size


##########
# t-test #
##########
def eval_t_test(
    estimate: Bound,
    std: float,
    num_of_elements: float,
    delta: float,
    two_sided: bool = True,
) -> tuple[Bound, Bound]:
    """Student's t interval. ``std`` is the unbiased sample standard deviation."""
    if num_of_elements < 2:
        return -math.inf, math.inf
    tail = delta / 2.0 if two_sided else delta
    t = float(stats.t.ppf(1 - tail, num_of_elements - 1))
    int_size = (std / math.sqrt(num_of_elements)) * t
    return estimate - int_size, estimate + int_size


def predict_t_test(
    estimate: Bound, std: float, safety_size: float, delta: float
) -> tuple[Bound, Bound]:
    """Candidate-selection prediction of the t-test safety interval."""
    if safety_size < 2:
        return -math.inf, math.inf
    t = float(stats.t.ppf(1 - delta / 2.0, safety_size - 1))
    int_size = 2 * (std / math.sqrt(safety_size)) * t
    return estimate - int_size, estimate + int_size


###########
# Betting #
###########
#: Bisection steps used to locate each endpoint once it has been bracketed. Each
#: step costs a pass over the data, and 24 halvings resolve a bracket of width 1
#: to about 6e-8 - far finer than any grid of comparable cost.
_BETTING_REFINE = 24


def _capital_rejects(
    x: "np.ndarray", m: float, log_threshold: float, delta: float
) -> bool:
    r"""Whether the betting capital process rejects the hypothesis ``mean = m``.

    Two nonnegative martingales are run against ``m``: one betting that the true
    mean exceeds ``m`` and one that it falls below. Under the hypothesis each has
    unit expectation, so by Ville's inequality each exceeds ``2/delta`` with
    probability at most ``delta/2``; rejecting when either does gives a level
    ``delta`` test, and inverting the test over ``m`` gives the interval.

    The bets ``lambda_i`` must be *predictable* - a function of
    :math:`x_1, \dots, x_{i-1}` only - or the martingale property fails and the
    guarantee with it. They are therefore computed from a running mean and
    variance that exclude the current observation.
    """
    n = x.size
    if n == 0:
        return False
    # Predictable bet size: the variance-adapted choice of Waudby-Smith and
    # Ramdas, truncated so that the capital stays strictly positive for any
    # observation in [0, 1].
    ceiling = 0.5 / max(m, 1.0 - m, 1e-12)
    running_sum = 0.0
    running_sq = 0.0
    log_k_up = 0.0
    log_k_down = 0.0
    scale = 2.0 * math.log(2.0 / delta)
    for i, value in enumerate(x):
        mean_prev = (running_sum + 0.5) / (i + 1)
        var_prev = max((running_sq + 0.25) / (i + 1) - mean_prev**2, 1e-4)
        bet = min(math.sqrt(scale / (var_prev * n)), ceiling)
        centred = float(value) - m
        log_k_up += math.log1p(bet * centred)
        log_k_down += math.log1p(-bet * centred)
        if log_k_up >= log_threshold or log_k_down >= log_threshold:
            return True
        running_sum += float(value)
        running_sq += float(value) ** 2
    return False


def betting_interval(
    x: "np.ndarray", delta: float, two_sided: bool = True
) -> tuple[Bound, Bound]:
    """Betting confidence interval for the mean of values in [0, 1].

    The interval is the set of candidate means the capital test does not reject.
    Rather than scan a grid - which can step straight over a narrow interval and
    find nothing - we anchor at the sample mean, walk outwards until the test
    rejects, and bisect. Each endpoint is reported on the rejected side of the
    boundary, so discretisation can only ever widen the interval.
    """
    if x.size == 0:
        return -math.inf, math.inf
    tail = delta / 2.0 if two_sided else delta
    log_threshold = math.log(2.0 / tail)

    def rejects(m: float) -> bool:
        return _capital_rejects(x, min(max(m, 0.0), 1.0), log_threshold, tail)

    centre = float(np.clip(np.mean(x), 1e-9, 1 - 1e-9))
    if rejects(centre):
        # The test rejects even the sample mean; claim nothing.
        return -math.inf, math.inf

    def walk(direction: int) -> float:
        limit = 1.0 if direction > 0 else 0.0
        span = 1e-3
        outside = centre
        while True:  # expand until the test rejects
            candidate = centre + direction * span
            if (direction > 0 and candidate >= limit) or (
                direction < 0 and candidate <= limit
            ):
                if not rejects(limit):
                    return limit  # interval runs to the boundary
                outside = limit
                break
            if rejects(candidate):
                outside = candidate
                break
            span *= 2.0
        # The sample mean is known not to be rejected, so it is always a valid
        # inner end of the bracket. Starting there rather than at the last
        # accepted candidate only widens the bracket, which the bisection below
        # closes anyway.
        inside = centre
        low, high = (inside, outside) if direction > 0 else (outside, inside)
        for _ in range(_BETTING_REFINE):  # bisect towards the boundary
            middle = 0.5 * (low + high)
            if rejects(middle):
                if direction > 0:
                    high = middle
                else:
                    low = middle
            elif direction > 0:
                low = middle
            else:
                high = middle
        return high if direction > 0 else low

    return walk(-1), walk(+1)
