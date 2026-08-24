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

from fair_seldonian.data.synthetic import get_data


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
