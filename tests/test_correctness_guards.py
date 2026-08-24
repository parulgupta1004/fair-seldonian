"""Tests for the failure modes that mostly do not announce themselves.

Nearly every test here pins down a choice whose wrong answer produces
plausible-looking numbers rather than an error: a coverage guarantee that is
quietly half what it claims, an objective that is not the one on the axis label,
a benchmark with no accuracy to trade away. Ordinary unit tests do not catch
those, because nothing raises and nothing looks absurd. The comments record why
each assertion is what it is, so that none of them is later "fixed" by relaxing
it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import log_loss

from fair_seldonian.constraints.inequalities import (
    Inequality,
    check_constraint_groups,
    eval_func_bound,
)
from fair_seldonian.data.synthetic import get_data
from fair_seldonian.models.logistic_regression import f_hat


# --------------------------------------------------------------------------
# Confidence-interval coverage
# --------------------------------------------------------------------------
def _empirical_coverage(
    n_total: int,
    n_minority: int,
    p_true: float,
    inequality: Inequality,
    trials: int = 300,
    delta: float = 0.05,
) -> float:
    rng = np.random.default_rng(7)
    T = pd.Series(np.array(["0"] * (n_total - n_minority) + ["1"] * n_minority))
    hits = 0
    for _ in range(trials):
        Y = np.zeros(n_total, dtype=int)
        Y[: n_total - n_minority] = rng.binomial(1, 0.5, n_total - n_minority)
        minority = rng.binomial(1, p_true, n_minority)
        Y[n_total - n_minority :] = minority
        pred = np.zeros(n_total)
        pred[n_total - n_minority :] = minority
        pred[: n_total - n_minority] = rng.random(n_total - n_minority)
        lo, hi = eval_func_bound(
            "TP(1)",
            pd.Series(Y),
            torch.tensor(pred),
            T,
            delta,
            inequality,
            None,
            False,
            False,
        )
        hits += lo <= p_true <= hi
    return hits / trials


def test_interval_covers_on_a_small_minority_group() -> None:
    """The distribution-free interval must cover a minority group's value.

    This is the assertion the whole guarantee rests on. Building the interval from
    ``#{Y == 1}`` across *all* groups while the estimate divides by the size of one
    group makes it about 4.5x too narrow on a 500-member group inside 20,000
    samples, and drops empirical coverage to roughly 0.5 against a nominal 0.95. A
    balanced benchmark cannot detect it, because there the two counts coincide
    numerically.
    """
    assert (
        _empirical_coverage(20_000, 500, 0.3, Inequality.HOEFFDING_INEQUALITY) >= 0.95
    )


def test_interval_width_scales_with_group_size_not_dataset_size() -> None:
    """Growing the majority around a fixed minority must not shrink its interval."""
    widths = []
    for n_total in (2_000, 20_000, 200_000):
        n_minority = 200
        T = pd.Series(np.array(["0"] * (n_total - n_minority) + ["1"] * n_minority))
        Y = pd.Series(np.ones(n_total, dtype=int))
        pred = torch.tensor(np.full(n_total, 0.5))
        lo, hi = eval_func_bound(
            "TP(1)",
            Y,
            pred,
            T,
            0.05,
            Inequality.HOEFFDING_INEQUALITY,
            None,
            False,
            False,
        )
        widths.append(hi - lo)
    assert max(widths) - min(widths) < 1e-12


def test_group_dtype_mismatch_raises_instead_of_failing_silently() -> None:
    """Float group labels make every mask empty and every bound +inf.

    That looks exactly like a legitimate "no solution found", so the run would
    report a clean negative result rather than a type error.
    """
    check_constraint_groups(["0", "1"], pd.Series([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="upcast|appear in no row"):
        check_constraint_groups(["0", "1"], np.array([0.0, 1.0, 0.0]))


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
def test_f_hat_is_log_loss() -> None:
    """``f_hat`` must be the negative log loss, not softmax-of-probabilities.

    Passing probabilities to ``CrossEntropyLoss`` (which expects logits) yields a
    quantity floored at 0.3133 for a perfect classifier, so any curve plotted from
    it asymptotes to 0.31 rather than towards the Bayes rate, and the range is
    compressed roughly tenfold.
    """
    Y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    for probabilities in (
        np.array([0.01, 0.99, 0.99, 0.01, 0.99, 0.01, 0.99, 0.99]),
        np.array([0.2, 0.8, 0.7, 0.3, 0.9, 0.1, 0.85, 0.75]),
        np.full(8, 0.5),
    ):
        # A single feature with unit weight and no intercept makes predict() the
        # identity on the supplied probabilities' logits.
        logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
        theta = torch.tensor(np.array([1.0]))
        theta1 = torch.tensor(np.array([0.0]))
        got = -float(f_hat(theta, theta1, logits, Y))
        assert abs(got - log_loss(Y, probabilities)) < 1e-9


def test_perfect_classifier_approaches_zero_loss() -> None:
    Y = np.array([0, 1, 1, 0])
    logits = np.where(Y == 1, 30.0, -30.0).reshape(-1, 1)
    loss = -float(
        f_hat(torch.tensor(np.array([1.0])), torch.tensor(np.array([0.0])), logits, Y)
    )
    assert loss < 1e-6


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def test_synthetic_problem_has_non_zero_bayes_error() -> None:
    """The signal feature must not determine the label.

    A construction like ``Y * uniform(0, 1)`` is zero exactly when ``Y = 0``, so
    the label is perfectly recoverable and there is no accuracy to trade away.
    """
    data = get_data(20_000, 5, 0.5, 0.4, 0.6, random_seed=0.5)
    signal = np.asarray(data.iloc[:, 0])
    Y = np.asarray(data.iloc[:, -2])
    best = max(
        np.mean((signal > threshold) == (Y == 1))
        for threshold in np.linspace(-1, 2, 61)
    )
    assert best < 0.95


def test_sensitive_feature_inclusion_is_explicit() -> None:
    with_t = get_data(2_000, 5, 0.5, 0.4, 0.6, 0.5, include_sensitive_feature=True)
    without_t = get_data(2_000, 5, 0.5, 0.4, 0.6, 0.5, include_sensitive_feature=False)
    assert np.array_equal(
        np.asarray(with_t.iloc[:, -3]), np.asarray(with_t.iloc[:, -1])
    )
    assert not np.array_equal(
        np.asarray(without_t.iloc[:, -3]), np.asarray(without_t.iloc[:, -1])
    )
    assert with_t.shape[1] == without_t.shape[1] == 7


def test_base_rate_gap_is_present() -> None:
    data = get_data(40_000, 5, 0.5, 0.4, 0.6, random_seed=0.5)
    Y = np.asarray(data.iloc[:, -2])
    T = np.asarray(data.iloc[:, -1])
    assert abs(Y[T == 0].mean() - 0.4) < 0.02
    assert abs(Y[T == 1].mean() - 0.6) < 0.02


def test_trials_with_different_seeds_are_independent_draws() -> None:
    """Distinct seeds must give distinct datasets.

    Drawing one dataset outside the trial loop and subsampling it makes the error
    bars measure subsampling noise, and at the largest size every trial sees
    identical data.
    """
    a = get_data(1_000, 5, 0.5, 0.4, 0.6, random_seed=0.1)
    b = get_data(1_000, 5, 0.5, 0.4, 0.6, random_seed=0.2)
    assert not np.array_equal(np.asarray(a.iloc[:, 0]), np.asarray(b.iloc[:, 0]))
