from __future__ import annotations

from random import random, seed
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(
    N: int,
    features: int,
    t_ratio: float,
    tp0_ratio: float,
    tp1_ratio: float,
    random_seed: float,
) -> pd.DataFrame:
    random_state = int(random_seed * 99) + 1
    seed(random_state)
    T = np.random.default_rng(random_state).binomial(1, t_ratio, N)
    A = np.zeros(T.shape)
    Y = np.zeros(T.shape)
    X = np.zeros(T.shape)
    group0_X = A[T.astype(str) == "0"]
    group0_Y = np.random.default_rng(random_state).binomial(
        1, tp0_ratio, group0_X.shape
    )
    group1_X = A[T.astype(str) == "1"]
    group1_Y = np.random.default_rng(random_state).binomial(
        1, tp1_ratio, group1_X.shape
    )
    j = 0  # for 0
    k = 0  # for 1
    for i in range(T.shape[0]):
        if T[i] == 0:
            # get from group0_Y
            Y[i] = group0_Y[j]
            X[i] = Y[i] * random()

            j += 1
        elif T[i] == 1:
            # get from group1_Y
            Y[i] = group1_Y[k]
            X[i] = Y[i] * random()
            k += 1

    T = pd.Series(T)
    # Use a seeded generator (not the global np.random) so the noise feature
    # columns are reproducible; otherwise get_data is nondeterministic across runs.
    extra_features = np.random.default_rng(random_state).random((N, features - 2))
    X = pd.concat([pd.DataFrame(X), pd.DataFrame(extra_features), T], axis=1)
    Y = pd.Series(Y)
    return pd.concat([X, Y, T], axis=1)


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
