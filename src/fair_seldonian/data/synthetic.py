from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#: Standard deviation of the Gaussian noise added to the label to form the signal
#: feature. The Bayes error of the resulting problem is ``Phi(-1 / (2 * sigma))``,
#: so the default of 0.5 gives about 16%: unconstrained logistic regression reaches
#: roughly 84% accuracy at a log loss near 0.35, which leaves a real accuracy cost
#: to pay when the fairness constraint is enforced.
DEFAULT_SIGNAL_NOISE = 0.5


def get_data(
    N: int,
    features: int,
    t_ratio: float,
    tp0_ratio: float,
    tp1_ratio: float,
    random_seed: float,
    signal_noise: float = DEFAULT_SIGNAL_NOISE,
    include_sensitive_feature: bool = True,
) -> pd.DataFrame:
    """Synthetic binary-classification data with a base-rate gap between groups.

    ``T ~ Bernoulli(t_ratio)`` selects the group; the label is then drawn with
    ``P(Y=1 | T=0) = tp0_ratio`` and ``P(Y=1 | T=1) = tp1_ratio``. The gap between
    those two base rates is what makes the fairness constraint bite.

    The signal feature is ``Y + N(0, signal_noise)``, giving a Bayes error of
    ``Phi(-1 / (2 * signal_noise))``. Additive noise is essential: a construction
    like ``Y * uniform(0, 1)`` is zero exactly when ``Y = 0`` and positive
    otherwise, so the label is perfectly recoverable, the Bayes error is zero, and
    there is no accuracy-fairness frontier to trade along - any loss the
    constrained model paid would be an artifact rather than a real cost.

    :param N: number of samples.
    :param features: total number of columns in the returned feature block,
        including the sensitive column when ``include_sensitive_feature`` is set.
    :param t_ratio: ``P(T = 1)``.
    :param tp0_ratio: ``P(Y = 1 | T = 0)``.
    :param tp1_ratio: ``P(Y = 1 | T = 1)``.
    :param random_seed: seed; distinct values give independent datasets.
    :param signal_noise: standard deviation of the label noise in the signal feature.
    :param include_sensitive_feature: whether ``T`` is also supplied to the model as
        an input feature ("fairness through awareness"). Explicit because it
        materially changes the problem: with ``T`` available the classifier can
        condition directly on group membership. Easy to leave unstated, too, since
        appending ``T`` to the feature block survives the column slicing in
        :func:`data_split` without comment.
    :return: frame of ``features + 2`` columns - the feature block, then ``Y``,
        then ``T``.
    """
    random_state = int(random_seed * 99) + 1
    # One generator for every draw. Re-seeding a fresh generator per draw with the
    # same value would couple T, Y and the noise columns to a common stream.
    rng = np.random.default_rng(random_state)

    T = rng.binomial(1, t_ratio, N)
    base_rate = np.where(T == 1, tp1_ratio, tp0_ratio)
    Y = rng.binomial(1, base_rate)

    signal = Y + rng.normal(0.0, signal_noise, N)
    n_noise = features - 2 if include_sensitive_feature else features - 1
    noise_cols = rng.normal(0.0, 1.0, (N, max(n_noise, 0)))

    blocks = [pd.DataFrame(signal), pd.DataFrame(noise_cols)]
    T = pd.Series(T)
    if include_sensitive_feature:
        blocks.append(T)
    X = pd.concat(blocks, axis=1)
    return pd.concat([X, pd.Series(Y.astype(float)), T], axis=1)


def data_split(
    frac: float, all_data: pd.DataFrame, random_state: int, m_test: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_train, all_test, Y_train, Y_test = cast(
        "tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]",
        train_test_split(
            all_data, all_data.iloc[:, -2], test_size=m_test, random_state=42
        ),
    )
    # test dataset
    T_test = all_test.iloc[:, -1]
    X_test = all_test.iloc[:, :-2]

    # train
    subsampling = all_train.sample(frac=frac, random_state=random_state)
    subsampling = subsampling.reset_index()
    subsampling = subsampling.drop(columns=["index"])
    T = subsampling.iloc[:, -1]
    X = subsampling.iloc[:, :-2]
    Y = subsampling.iloc[:, -2]
    return (
        np.array(X_test),
        np.array(Y_test),
        np.array(T_test),
        np.array(X),
        np.array(Y),
        np.array(T),
    )
