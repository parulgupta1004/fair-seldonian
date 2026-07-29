"""Tests for the ready-to-use fairness constraint builders."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from fair_seldonian import (
    FAIRNESS_CONSTRAINTS,
    demographic_parity,
    equal_opportunity,
    equalized_odds,
)
from fair_seldonian.constraints import construct_expr_tree_base, eval_expr_tree_base


def _dataset() -> tuple[pd.Series, torch.Tensor, pd.Series]:
    """A hand-checkable 2-group dataset with 0/1 predictions.

    Per-group confusion cells (each divided by the group size of 4):

    ==========  ====  ====  ====  ====  =====  =====  =====
    group       TP    FP    FN    TN    PPR    TPR    FPR
    ==========  ====  ====  ====  ====  =====  =====  =====
    "1"         .25   .25   .25   .25   .50    .50    .50
    "0"         .50   .00   .00   .50   .50    1.0    .00
    ==========  ====  ====  ====  ====  =====  =====  =====
    """
    T = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    Y = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
    pred = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    return Y, pred, T


def _evaluate(constraint: str) -> float:
    Y, pred, T = _dataset()
    tree = construct_expr_tree_base(constraint)
    value = eval_expr_tree_base(tree, Y, pred, T)
    assert value is not None
    return float(value)


def test_demographic_parity_matches_notebook_string() -> None:
    # The Adult notebook hand-writes exactly this postfix string.
    assert demographic_parity(0.1) == "TP(1) FP(1) + TP(0) FP(0) + - abs 0.1 -"


def test_demographic_parity_value() -> None:
    # |PPR(1) - PPR(0)| - eps = |0.50 - 0.50| - 0.10 = -0.10
    assert _evaluate(demographic_parity(0.1)) == pytest.approx(-0.10)


def test_equal_opportunity_value() -> None:
    # |TPR(1) - TPR(0)| - eps = |0.50 - 1.00| - 0.10 = 0.40
    assert _evaluate(equal_opportunity(0.1)) == pytest.approx(0.40)


def test_equalized_odds_value() -> None:
    # |TPR gap| + |FPR gap| - eps = 0.50 + |0.50 - 0.00| - 0.10 = 0.90
    assert _evaluate(equalized_odds(0.1)) == pytest.approx(0.90)


def test_all_constraints_parse() -> None:
    for build in FAIRNESS_CONSTRAINTS.values():
        # Should construct a tree without raising.
        assert construct_expr_tree_base(build(0.1)) is not None


def test_custom_group_labels() -> None:
    c = demographic_parity(0.05, groups=("Male", "Female"))
    assert c == "TP(Male) FP(Male) + TP(Female) FP(Female) + - abs 0.05 -"


def test_integer_group_labels_are_stringified() -> None:
    assert equal_opportunity(0.1, groups=(1, 0)) == equal_opportunity(0.1)


@pytest.mark.parametrize("build", list(FAIRNESS_CONSTRAINTS.values()))
def test_rejects_negative_epsilon(build) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        build(-0.01)


@pytest.mark.parametrize("build", list(FAIRNESS_CONSTRAINTS.values()))
def test_rejects_duplicate_groups(build) -> None:
    with pytest.raises(ValueError, match="distinct"):
        build(0.1, groups=("1", "1"))


@pytest.mark.parametrize("build", list(FAIRNESS_CONSTRAINTS.values()))
def test_rejects_malformed_group_label(build) -> None:
    with pytest.raises(ValueError, match="whitespace or parentheses"):
        build(0.1, groups=("a b", "0"))


def test_rejects_empty_group_label() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        demographic_parity(0.1, groups=("", "0"))


def test_rejects_wrong_group_count() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        demographic_parity(0.1, groups=("1", "0", "2"))  # type: ignore[arg-type]
