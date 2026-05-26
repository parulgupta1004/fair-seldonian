import torch
from fair_seldonian.constraints.inequalities import (
    Inequality, eval_estimate, eval_hoeffding, predict_hoeffding,
    predict_hoeffding_modified, eval_t_test, get_variance, get_num_of_elements,
)
from fair_seldonian.data.synthetic import get_data


def _data(n=200):
    d = get_data(N=n, features=3, t_ratio=0.5, tp0_ratio=0.5, tp1_ratio=0.5, random_seed=7)
    return d.iloc[:, -2], torch.tensor(d.iloc[:, -2].values, dtype=torch.float64), d.iloc[:, -1]


def test_hoeffding_symmetric():
    lo, hi = eval_hoeffding(0.5, 100, 0.05)
    assert abs((0.5 - lo) - (hi - 0.5)) < 1e-10

def test_hoeffding_shrinks_with_data():
    _, hi100 = eval_hoeffding(0.5, 100, 0.05)
    _, hi1000 = eval_hoeffding(0.5, 1000, 0.05)
    assert hi1000 < hi100

def test_predict_wider_than_eval():
    _, hi_pred = predict_hoeffding(0.5, 100, 0.05)
    _, hi_eval = eval_hoeffding(0.5, 100, 0.05)
    assert hi_pred > hi_eval

def test_modified_tighter():
    _, hi_std = predict_hoeffding(0.5, 100, 0.05)
    _, hi_mod = predict_hoeffding_modified(0.5, 200, 100, 0.05)
    assert hi_mod < hi_std

def test_ttest_zero_variance():
    lo, hi = eval_t_test(0.5, 0.0, 100, 0.05)
    assert lo == 0.5 and hi == 0.5

def test_estimate_range():
    Y, pred, T = _data()
    for func in ["TP(1)", "TN(0)", "FP(1)", "FN(0)"]:
        assert 0 <= float(eval_estimate(func, Y, pred, T)) <= 1

def test_variance_nonneg():
    Y, pred, T = _data()
    est = eval_estimate("TP(1)", Y, pred, T)
    n = get_num_of_elements("TP(1)", Y)
    assert get_variance("TP(1)", est, pred, T, n) >= 0
