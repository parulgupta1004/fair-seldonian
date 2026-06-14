import pandas as pd
import torch

from fair_seldonian.constraints.expression_tree_ext import (
    ExprTree,
    check_node_dup,
    construct_expr_tree,
    eval_expr_tree_conf_interval,
    is_constant,
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


def test_is_constant() -> None:
    assert is_constant("5") and is_constant("0.25") and is_constant("-3.14")
    assert not is_constant("TP(1)") and not is_constant("+")


def test_uniform_delta() -> None:
    t = construct_expr_tree("TP(1) TP(0) -", 0.1, False, False)
    assert t.left is not None and t.right is not None
    assert abs(t.left.delta - 0.05) < 1e-10
    assert abs(t.right.delta - 0.05) < 1e-10


def test_constant_aware_delta() -> None:
    t = construct_expr_tree("TP(1) 0.25 -", 0.1, False, True)
    assert t.left is not None
    assert abs(t.left.delta - 0.1) < 1e-10


def test_duplicate_detection() -> None:
    t = construct_expr_tree("TP(1) TP(0) - abs 0.25 TP(1) * -", 0.1, False, False)
    h: dict[str, list[float]] = {}
    check_node_dup(t, h)
    assert len(h["TP(1)"]) == 2


def test_bound_unifies_deltas() -> None:
    t = construct_expr_tree("TP(1) TP(0) - abs 0.25 TP(1) * -", 0.1, True, False)
    deltas: list[float] = []

    def collect(n: ExprTree | None) -> None:
        if n:
            collect(n.left)
            if n.value == "TP(1)":
                deltas.append(n.delta)
            collect(n.right)

    collect(t)
    assert len(deltas) == 2 and deltas[0] == deltas[1]


def test_optimizations_tighten() -> None:
    Y, pred, T = _data()
    expr = "TP(1) TP(0) - abs 0.25 TP(1) * -"
    _, hi_base = eval_expr_tree_conf_interval(
        construct_expr_tree(expr, 0.05, False, False),
        Y,
        pred,
        T,
        Inequality.HOEFFDING_INEQUALITY,
        1,
        True,
        False,
    )
    _, hi_opt = eval_expr_tree_conf_interval(
        construct_expr_tree(expr, 0.05, True, True),
        Y,
        pred,
        T,
        Inequality.HOEFFDING_INEQUALITY,
        1,
        True,
        True,
    )
    assert hi_base is not None and hi_opt is not None
    assert float(hi_opt) <= float(hi_base)
