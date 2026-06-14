import math
import threading
from typing import cast

import pandas as pd
import pytest
import torch

from fair_seldonian.constraints.inequalities import (
    Inequality,
    eval_estimate,
    eval_func_bound,
    eval_hoeffding,
    eval_t_test,
    get_num_of_elements,
    get_variance,
    group_mask,
    predict_hoeffding,
    predict_hoeffding_modified,
)
from fair_seldonian.data.synthetic import get_data


def _data(n: int = 200) -> tuple[pd.Series, torch.Tensor, pd.Series]:
    d = get_data(
        N=n, features=3, t_ratio=0.5, tp0_ratio=0.5, tp1_ratio=0.5, random_seed=7
    )
    return (
        d.iloc[:, -2],
        torch.tensor(d.iloc[:, -2].values, dtype=torch.float64),
        d.iloc[:, -1],
    )


def test_hoeffding_symmetric() -> None:
    lo, hi = eval_hoeffding(0.5, 100, 0.05)
    assert abs((0.5 - lo) - (hi - 0.5)) < 1e-10


def test_hoeffding_shrinks_with_data() -> None:
    _, hi100 = eval_hoeffding(0.5, 100, 0.05)
    _, hi1000 = eval_hoeffding(0.5, 1000, 0.05)
    assert hi1000 < hi100


def test_predict_wider_than_eval() -> None:
    _, hi_pred = predict_hoeffding(0.5, 100, 0.05)
    _, hi_eval = eval_hoeffding(0.5, 100, 0.05)
    assert hi_pred > hi_eval


def test_modified_tighter() -> None:
    _, hi_std = predict_hoeffding(0.5, 100, 0.05)
    _, hi_mod = predict_hoeffding_modified(0.5, 200, 100, 0.05)
    assert hi_mod < hi_std


def test_ttest_zero_variance() -> None:
    lo, hi = eval_t_test(0.5, 0.0, 100, 0.05)
    assert lo == 0.5 and hi == 0.5


def test_estimate_range() -> None:
    Y, pred, T = _data()
    for func in ["TP(1)", "TN(0)", "FP(1)", "FN(0)"]:
        assert 0 <= float(eval_estimate(func, Y, pred, T)) <= 1


def test_variance_nonneg() -> None:
    Y, pred, T = _data()
    est = eval_estimate("TP(1)", Y, pred, T)
    n = get_num_of_elements("TP(1)", Y)
    assert get_variance("TP(1)", est, pred, T, n) >= 0


def test_estimate_unknown_variable_raises() -> None:
    Y, pred, T = _data()
    with pytest.raises(ValueError):
        eval_estimate("XX(1)", Y, pred, T)


def test_func_bound_unknown_inequality_raises() -> None:
    Y, pred, T = _data()
    with pytest.raises(ValueError):
        # Intentionally pass an invalid inequality to exercise the error path.
        eval_func_bound(
            "TP(1)", Y, pred, T, 0.05, cast("Inequality", "bogus"), None, False, False
        )


def test_num_of_elements_unknown_variable_raises() -> None:
    Y, _, _ = _data()
    with pytest.raises(ValueError):
        get_num_of_elements("XX(1)", Y)


def test_estimate_empty_group_returns_zero_not_nan() -> None:
    # No rows belong to group "1": the rate is undefined but must not be NaN.
    Y = pd.Series([1, 0, 1, 0])
    T = pd.Series([0, 0, 0, 0])
    pred = torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float64)
    est = float(eval_estimate("TP(1)", Y, pred, T))
    assert est == 0.0 and not math.isnan(est)


def test_func_bound_empty_group_fails_closed() -> None:
    # Empty group -> widest interval so the safety test fails closed (no div-by-zero).
    Y = pd.Series([1, 0, 1, 0])
    T = pd.Series([0, 0, 0, 0])
    pred = torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float64)
    for inequality in (Inequality.HOEFFDING_INEQUALITY, Inequality.T_TEST):
        lo, hi = eval_func_bound(
            "TP(1)", Y, pred, T, 0.05, inequality, None, False, False
        )
        assert lo == -math.inf and hi == math.inf


def test_func_bound_single_sample_ttest_fails_closed() -> None:
    # Only one positive label -> t-test cannot form an interval (df=0); fail closed.
    Y = pd.Series([1, 0, 0, 0])
    T = pd.Series([1, 1, 1, 1])
    pred = torch.tensor([0.9, 0.1, 0.2, 0.3], dtype=torch.float64)
    lo, hi = eval_func_bound(
        "TP(1)", Y, pred, T, 0.05, Inequality.T_TEST, None, False, False
    )
    assert lo == -math.inf and hi == math.inf


def test_group_mask_thread_safe() -> None:
    # The group-mask cache is shared module state. On the free-threaded (PEP 703)
    # build these calls run in true parallel, so concurrent lookups, inserts, and
    # evictions must stay correct without corrupting the cache.
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()  # release all threads together to maximise contention
        try:
            for _ in range(200):
                # A fresh Series each iteration yields a new id(), forcing a
                # cache miss and exercising the lock-guarded insert/clear path.
                t = pd.Series([0, 1, 1, 0, 1])
                mask = group_mask(t, "1")
                assert mask.tolist() == [False, True, True, False, True]
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
