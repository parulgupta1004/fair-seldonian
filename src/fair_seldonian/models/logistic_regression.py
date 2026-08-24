from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ..config import DEFAULT_CONFIG, SeldonianConfig
from ..constraints.affine import affine_upper_bound
from ..constraints.expression_tree import (
    construct_expr_tree_base,
    eval_expr_tree_conf_interval_base,
)
from ..constraints.expression_tree_ext import (
    construct_expr_tree,
    eval_expr_tree_conf_interval,
)

if TYPE_CHECKING:
    from .._typing import Array, Bound

#: The names this module contributes to the public API. autodoc documents
#: exactly these, so the API reference stays the surface users are meant to
#: call rather than every helper that happens to lack a leading underscore.
__all__ = [
    "eval_ghat",
    "f_hat",
    "ghat",
    "predict",
    "simple_logistic",
]

logger = logging.getLogger(__name__)


def predict(
    theta: torch.Tensor | None, theta1: torch.Tensor | None, X: np.ndarray
) -> torch.Tensor:
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


# Keeps log(p) finite when the sigmoid saturates to exactly 0 or 1 in float64.
_EPS = 1e-12


def f_hat(
    theta: torch.Tensor | None, theta1: torch.Tensor | None, X: np.ndarray, Y: Array
) -> torch.Tensor:
    """
    Main objective function: negative log loss (higher is better).

    Note that ``torch.nn.CrossEntropyLoss`` is the wrong tool here. It expects raw
    *logits* and applies ``log_softmax`` internally, so stacking the predicted
    probabilities into a two-column tensor and passing them to it computes
    ``-log softmax([1-p, p])_y`` - a different function, with a floor of 0.3133 for
    a perfect classifier and a ceiling near 1.31. That both mislabels any axis
    called "log loss" and compresses the range roughly tenfold, which destroys the
    resolution of small between-method differences.

    :param theta: The optimal theta values for the model
    :param theta1: The additional optimal theta values for the model
    :param X: The features of the dataset
    :param Y: The true labels of the dataset
    :return: The negative log loss
    """
    pred = predict(theta, theta1, X).clamp(_EPS, 1 - _EPS)
    target = torch.tensor(np.asarray(Y), dtype=pred.dtype)
    return -torch.nn.functional.binary_cross_entropy(pred, target)


def simple_logistic(X: np.ndarray, Y: Array) -> tuple[torch.Tensor, torch.Tensor]:
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
            theta1,
            requires_grad=True,
        ), torch.tensor(np.array([theta0]), requires_grad=True)
    except Exception:
        logger.exception("Exception in logRes")
        raise


@dataclass(frozen=True)
class SeldonianType:
    """How one ``seldonian_type`` composes its per-leaf intervals into a bound.

    ``summary`` is rendered directly into the documentation, so a new variant is
    described in exactly one place. ``family`` selects the implementation and the
    remaining fields are the flags it takes.
    """

    summary: str
    family: str  # "base" | "extend" | "affine"
    modified_h: bool = False
    check_bound: bool = False
    check_const: bool = False


#: Every accepted ``seldonian_type``, in the order they build on one another. This
#: is the single source of truth: :func:`eval_ghat` and :func:`ghat` dispatch on it,
#: the error message for an unknown type enumerates it, and the documentation's
#: variant table is generated from it. Adding a variant means adding one entry.
SELDONIAN_TYPES: dict[str, SeldonianType] = {
    "base": SeldonianType(
        "Uniform delta/2 splitting at every operator, with the standard Hoeffding "
        "bound predicting the safety test.",
        family="base",
    ),
    "mod": SeldonianType(
        "Decomposes candidate and safety estimation error instead of doubling the "
        "safety term, which is tighter whenever the two splits differ in size.",
        family="base",
        modified_h=True,
    ),
    "const": SeldonianType(
        "Passes the full delta to the variable subtree when the sibling is a "
        "constant, since an exact value needs no interval.",
        family="extend",
        check_const=True,
    ),
    "bound": SeldonianType(
        "Merges the delta slices of repeated leaves into a single interval by the "
        "union bound, so a variable used three times is not paid for three times.",
        family="extend",
        check_bound=True,
    ),
    "opt": SeldonianType(
        "Combines mod, const and bound.",
        family="extend",
        modified_h=True,
        check_bound=True,
        check_const=True,
    ),
    "affine": SeldonianType(
        "Compiles the constraint to a max of affine forms and bounds each with one "
        "interval, exploiting independence across groups. Roughly halves the slack, "
        "but only for constraints built from +, -, scaling and abs.",
        family="affine",
    ),
}


def resolve_seldonian_type(seldonian_type: str) -> SeldonianType:
    """Look up a variant, naming the alternatives when it is not recognised."""
    try:
        return SELDONIAN_TYPES[seldonian_type]
    except KeyError:
        raise ValueError(
            f"Unknown seldonian_type: {seldonian_type!r}; "
            f"expected one of {', '.join(SELDONIAN_TYPES)}"
        ) from None


def eval_ghat(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    X: np.ndarray,
    Y: Array,
    T: Array,
    seldonian_type: str,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> Bound:
    spec = resolve_seldonian_type(seldonian_type)
    if spec.family == "base":
        bound = eval_ghat_base(
            theta, theta1, X, Y, T, modified_h=spec.modified_h, config=config
        )
    elif spec.family == "extend":
        bound = eval_ghat_extend(
            theta,
            theta1,
            X,
            Y,
            T,
            check_bound=spec.check_bound,
            check_const=spec.check_const,
            modified_h=spec.modified_h,
            config=config,
        )
    else:
        bound = ghat_affine(theta, theta1, X, Y, T, None, config)
    # A bound is a number, not a node in an autograd graph. simple_logistic
    # returns parameters with requires_grad set and predict propagates that, so
    # without this every `float(eval_ghat(...))` warns about converting a tensor
    # that requires grad. Nothing here differentiates through a bound - candidate
    # selection uses derivative-free Powell - so detach at the public boundary.
    return bound.detach() if isinstance(bound, torch.Tensor) else bound


def ghat(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    X: np.ndarray,
    Y: Array,
    T: Array,
    candidate_ratio: float,
    seldonian_type: str,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> Bound:
    spec = resolve_seldonian_type(seldonian_type)
    if spec.family == "base":
        return ghat_base(
            theta,
            theta1,
            X,
            Y,
            T,
            predict_bound=True,
            candidate_ratio=candidate_ratio,
            modified_h=spec.modified_h,
            config=config,
        )
    if spec.family == "extend":
        return ghat_extend(
            theta,
            theta1,
            X,
            Y,
            T,
            predict_bound=True,
            candidate_ratio=candidate_ratio,
            check_bound=spec.check_bound,
            check_const=spec.check_const,
            modified_h=spec.modified_h,
            config=config,
        )
    return ghat_affine(theta, theta1, X, Y, T, candidate_ratio, config)


def ghat_base(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    X: np.ndarray,
    Y: Array,
    T: Array,
    predict_bound: bool,
    candidate_ratio: float | None,
    modified_h: bool,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> Bound:
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
    if u is None:
        raise ValueError(
            f"Constraint {config.constraint!r} produced an undefined bound"
        )
    return u


def eval_ghat_base(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    X: np.ndarray,
    Y: Array,
    T: Array,
    modified_h: bool,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> Bound:
    return ghat_base(theta, theta1, X, Y, T, False, None, modified_h, config)


def ghat_extend(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    X: np.ndarray,
    Y: Array,
    T: Array,
    predict_bound: bool,
    candidate_ratio: float | None,
    check_bound: bool,
    check_const: bool,
    modified_h: bool,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> Bound:
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
    if u is None:
        raise ValueError(
            f"Constraint {config.constraint!r} produced an undefined bound"
        )
    return u


def eval_ghat_extend(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    X: np.ndarray,
    Y: Array,
    T: Array,
    check_bound: bool,
    check_const: bool,
    modified_h: bool,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> Bound:
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


def ghat_affine(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    X: np.ndarray,
    Y: Array,
    T: Array,
    candidate_ratio: float | None,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> Bound:
    """Upper bound via the max-of-affine-forms compilation.

    See :mod:`fair_seldonian.constraints.affine`. When ``candidate_ratio`` is given
    this is the candidate-selection prediction of the safety bound rather than the
    bound itself.
    """
    pred = predict(theta, theta1, X)
    root = construct_expr_tree_base(config.constraint)
    if candidate_ratio:
        sample_scale = (1 - candidate_ratio) / candidate_ratio
        inflate = 2.0
    else:
        sample_scale, inflate = 1.0, 1.0
    return affine_upper_bound(
        root, Y, pred, T, config.delta, sample_scale, inflate, config.inequality
    )
