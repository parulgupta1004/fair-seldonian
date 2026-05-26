import torch

from fair_seldonian.constraints.expression_tree import (
    construct_expr_tree_base,
    eval_expr_tree_base,
    eval_expr_tree_conf_interval_base,
    isFunc,
    isMod,
    isOperator,
)
from fair_seldonian.constraints.inequalities import Inequality
from fair_seldonian.data.synthetic import get_data


def _data(n=200):
    d = get_data(
        N=n, features=3, t_ratio=0.5, tp0_ratio=0.5, tp1_ratio=0.5, random_seed=7
    )
    return (
        d.iloc[:, -2],
        torch.tensor(d.iloc[:, -2].values, dtype=torch.float64),
        d.iloc[:, -1],
    )


def test_operators():
    for op in ["+", "-", "*", "/", "^"]:
        assert isOperator(op)
    assert not isOperator("abs")


def test_mod():
    assert isMod("abs") and not isMod("+")


def test_func():
    assert isFunc("TP(1)") and isFunc("FP(0)")
    assert not isFunc("abs") and not isFunc("0.5")


def test_simple_tree():
    t = construct_expr_tree_base("TP(1) TP(0) -")
    assert t.value == "-" and t.left.value == "TP(1)" and t.right.value == "TP(0)"


def test_abs_tree():
    t = construct_expr_tree_base("TP(1) TP(0) - abs")
    assert t.value == "abs" and t.right is None


def test_complex_tree():
    t = construct_expr_tree_base("TP(1) TP(0) - abs 0.25 TP(1) * -")
    assert t.value == "-" and t.left.value == "abs" and t.right.value == "*"


def test_eval():
    Y, pred, T = _data()
    assert (
        float(
            eval_expr_tree_base(construct_expr_tree_base("TP(1) TP(0) -"), Y, pred, T)
        )
        is not None
    )


def test_eval_abs():
    Y, pred, T = _data()
    assert (
        float(
            eval_expr_tree_base(
                construct_expr_tree_base("TP(1) TP(0) - abs"), Y, pred, T
            )
        )
        >= 0
    )


def test_eval_constant():
    Y, pred, T = _data()
    assert eval_expr_tree_base(construct_expr_tree_base("0.5"), Y, pred, T) == 0.5


def test_hoeffding_interval():
    Y, pred, T = _data()
    lo, hi = eval_expr_tree_conf_interval_base(
        construct_expr_tree_base("TP(1) TP(0) -"),
        Y,
        pred,
        T,
        0.05,
        Inequality.HOEFFDING_INEQUALITY,
        1,
        True,
        False,
    )
    assert float(lo) <= float(hi)


def test_ttest_interval():
    Y, pred, T = _data()
    lo, hi = eval_expr_tree_conf_interval_base(
        construct_expr_tree_base("TP(1) TP(0) -"),
        Y,
        pred,
        T,
        0.05,
        Inequality.T_TEST,
        1,
        True,
        False,
    )
    assert float(lo) <= float(hi)


def test_smaller_delta_widens():
    Y, pred, T = _data()
    tree = construct_expr_tree_base("TP(1)")
    _, hi_strict = eval_expr_tree_conf_interval_base(
        tree, Y, pred, T, 0.01, Inequality.HOEFFDING_INEQUALITY, 1, True, False
    )
    _, hi_loose = eval_expr_tree_conf_interval_base(
        tree, Y, pred, T, 0.10, Inequality.HOEFFDING_INEQUALITY, 1, True, False
    )
    assert float(hi_strict) >= float(hi_loose)
