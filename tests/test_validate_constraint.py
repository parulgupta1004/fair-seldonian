"""Tests for the postfix constraint validator used by ``SeldonianConfig``."""

from __future__ import annotations

import pytest

from fair_seldonian import (
    demographic_parity,
    equal_opportunity,
    equalized_odds,
    error_rate,
    error_rate_parity,
)
from fair_seldonian.constraints import validate_constraint


@pytest.mark.parametrize(
    "expr",
    [
        "TP(1)",  # single group rate
        "0.5",  # single constant
        "TP(1) TP(0) -",  # binary operator
        "TP(1) TP(0) - abs",  # abs (unary)
        "TP(1) 2 ^",  # power operator
        "TP(1) TP(1) FN(1) + /",  # division
        "FP(Male) FP(Female) - abs 0.05 -",  # multi-character group labels
        "TP(group_a) TP(group_b) -",  # underscore labels
    ],
)
def test_valid_constraints_pass(expr: str) -> None:
    assert validate_constraint(expr) is None


@pytest.mark.parametrize(
    "build",
    [demographic_parity, equal_opportunity, equalized_odds, error_rate_parity],
)
def test_builder_outputs_are_valid(build) -> None:
    assert validate_constraint(build(0.1)) is None


def test_error_rate_builder_output_is_valid() -> None:
    assert validate_constraint(error_rate(0.1)) is None


@pytest.mark.parametrize(
    ("expr", "match"),
    [
        ("", "non-empty"),
        ("   ", "non-empty"),
        ("TP(1)  TP(0) -", "empty token"),  # double space -> empty token
        ("TP(1) +", "two operands"),  # operator underflow
        ("abs", "one operand"),  # abs underflow
        ("TP(1) TP(0)", "does not reduce"),  # leftover operands
        ("foo", "unrecognized token"),  # not a number/operator/rate
        ("TP()", "unrecognized token"),  # empty group label -> not a valid rate
        ("SP(1)", "unrecognized token"),  # unknown rate prefix
    ],
)
def test_invalid_constraints_raise(expr: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_constraint(expr)
