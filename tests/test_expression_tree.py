import pandas as pd
import torch

from fair_seldonian.constraints.expression_tree import (
    construct_expr_tree_base,
    eval_expr_tree_base,
    eval_expr_tree_conf_interval_base,
    is_func,
    is_mod,
    is_operator,
)
from fair_seldonian.constraints.inequalities import Inequality
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


def test_operators() -> None:
    for op in ["+", "-", "*", "/", "^"]:
        assert is_operator(op)
    assert not is_operator("abs")


def test_mod() -> None:
    assert is_mod("abs") and not is_mod("+")


def test_func() -> None:
    assert is_func("TP(1)") and is_func("FP(0)")
    assert not is_func("abs") and not is_func("0.5")


def test_simple_tree() -> None:
    t = construct_expr_tree_base("TP(1) TP(0) -")
    assert t.left is not None and t.right is not None
    assert t.value == "-" and t.left.value == "TP(1)" and t.right.value == "TP(0)"


def test_abs_tree() -> None:
    t = construct_expr_tree_base("TP(1) TP(0) - abs")
    assert t.value == "abs" and t.right is None


def test_complex_tree() -> None:
    t = construct_expr_tree_base("TP(1) TP(0) - abs 0.25 TP(1) * -")
    assert t.left is not None and t.right is not None
    assert t.value == "-" and t.left.value == "abs" and t.right.value == "*"


def test_eval() -> None:
    Y, pred, T = _data()
    result = eval_expr_tree_base(construct_expr_tree_base("TP(1) TP(0) -"), Y, pred, T)
    assert result is not None and float(result) is not None


def test_eval_abs() -> None:
    Y, pred, T = _data()
    result = eval_expr_tree_base(
        construct_expr_tree_base("TP(1) TP(0) - abs"), Y, pred, T
    )
    assert result is not None and float(result) >= 0


def test_eval_constant() -> None:
    Y, pred, T = _data()
    assert eval_expr_tree_base(construct_expr_tree_base("0.5"), Y, pred, T) == 0.5


def test_hoeffding_interval() -> None:
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
    assert lo is not None and hi is not None
    assert float(lo) <= float(hi)


def test_ttest_interval() -> None:
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
    assert lo is not None and hi is not None
    assert float(lo) <= float(hi)


def test_smaller_delta_widens() -> None:
    Y, pred, T = _data()
    tree = construct_expr_tree_base("TP(1)")
    _, hi_strict = eval_expr_tree_conf_interval_base(
        tree, Y, pred, T, 0.01, Inequality.HOEFFDING_INEQUALITY, 1, True, False
    )
    _, hi_loose = eval_expr_tree_conf_interval_base(
        tree, Y, pred, T, 0.10, Inequality.HOEFFDING_INEQUALITY, 1, True, False
    )
    assert hi_strict is not None and hi_loose is not None
    assert float(hi_strict) >= float(hi_loose)
