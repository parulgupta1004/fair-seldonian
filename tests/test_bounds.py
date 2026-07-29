import math

import pytest

from fair_seldonian.constraints.bounds import (
    eval_abs_bound,
    eval_add_bound,
    eval_div_bound,
    eval_math_bound,
    eval_multiply_bound,
    eval_subtract_bound,
)

INF = math.inf


def test_add() -> None:
    assert eval_add_bound(1, 3, 2, 4) == (3, 7)


def test_subtract() -> None:
    assert eval_subtract_bound(1, 3, 2, 4) == (-3, 1)


def test_multiply() -> None:
    assert eval_multiply_bound(2, 3, 4, 5) == (8, 15)


def test_divide() -> None:
    lo, u = eval_div_bound(2, 6, 1, 3)
    assert lo is not None and u is not None
    assert abs(lo - 2 / 3) < 1e-10 and abs(u - 6) < 1e-10


def test_abs_positive() -> None:
    assert eval_abs_bound(2, 5) == (2, 5)


def test_abs_negative() -> None:
    assert eval_abs_bound(-5, -2) == (2, 5)


def test_abs_straddles_zero() -> None:
    assert eval_abs_bound(-7, 3) == (0, 7)


def test_inf() -> None:
    assert eval_add_bound(-math.inf, 3, 1, 2) == (-math.inf, 5)
    assert eval_div_bound(1, 2, -1, 1) == (-math.inf, math.inf)


def test_none() -> None:
    assert eval_add_bound(None, 1, 2, 3) == (None, None)


def test_dispatch() -> None:
    assert eval_math_bound(1, 3, 2, 4, "+") == (3, 7)
    assert eval_math_bound(-3, 5, None, None, "abs") == (0, 5)
    assert eval_math_bound(1, 2, 3, 4, "%") == (None, None)


def test_dispatch_subtract_multiply_divide_power() -> None:
    assert eval_math_bound(1, 2, 3, 4, "-") == (-3, -1)
    assert eval_math_bound(1, 2, 3, 4, "*") == (3, 8)
    assert eval_math_bound(1, 2, 2, 4, "/") == (0.25, 1.0)
    # power is intentionally not propagated: returns the left interval unchanged.
    assert eval_math_bound(1, 2, 3, 4, "^") == (1, 2)


def test_abs_infinite_endpoint() -> None:
    assert eval_abs_bound(-INF, 2) == (0, INF)
    assert eval_abs_bound(2, INF) == (0, INF)


def test_add_subtract_infinite_endpoints() -> None:
    assert eval_add_bound(1, 2, 3, INF) == (4, INF)
    assert eval_subtract_bound(-INF, 2, 3, 4) == (-INF, -1)
    assert eval_subtract_bound(1, 2, 3, INF) == (-INF, -1)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((1, 2, 3, 4), (3, 8)),  # pos * pos
        ((-2, -1, -4, -3), (3, 8)),  # neg * neg
        ((1, 2, -4, -3), (-8, -3)),  # pos * neg
        ((-2, -1, 3, 4), (-8, -3)),  # neg * pos
        ((-1, 2, 3, 4), (-4, 8)),  # 0 in x, y pos
        ((-1, 2, -4, -3), (-8, 4)),  # 0 in x, y neg
        ((1, 2, -3, 4), (-6, 8)),  # x pos, 0 in y
        ((-2, -1, -3, 4), (-8, 6)),  # x neg, 0 in y
        ((-1, 2, -3, 4), (-6, 8)),  # 0 in both
        ((-INF, INF, 1, 2), (-INF, INF)),  # unbounded factor
    ],
)
def test_multiply_sign_cases(args, expected) -> None:
    assert eval_multiply_bound(*args) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((1, 2, -1, 1), (-INF, INF)),  # denominator spans 0
        ((1, 2, 2, 4), (0.25, 1.0)),  # pos / pos
        ((-2, -1, -4, -2), (0.25, 1.0)),  # neg / neg
        ((1, 2, -4, -2), (-0.5, -0.25)),  # pos / neg
        ((-2, -1, 2, 4), (-1.0, -0.25)),  # neg / pos
        ((-1, 2, 2, 4), (-0.5, 1.0)),  # 0 in x, y pos
        ((-1, 2, -4, -2), (-1.0, 0.5)),  # 0 in x, y neg
    ],
)
def test_div_sign_cases(args, expected) -> None:
    lo, hi = eval_div_bound(*args)
    assert lo == pytest.approx(expected[0])
    assert hi == pytest.approx(expected[1])


def test_div_infinite_endpoints() -> None:
    assert eval_div_bound(1, 2, 2, INF) == (0, 1.0)  # u_y = inf -> lower 0
    assert eval_div_bound(1, INF, 2, 4) == (0.25, INF)  # u_x = inf -> upper inf


def test_none_operands_return_none() -> None:
    assert eval_multiply_bound(None, 1, 2, 3) == (None, None)
    assert eval_div_bound(None, 1, 2, 3) == (None, None)
    assert eval_subtract_bound(1, None, 2, 3) == (None, None)
    assert eval_abs_bound(None, None) == (None, None)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((1, INF, 3, 4), (3, INF)),  # pos * pos, u_x = inf
        ((-INF, -1, -4, -3), (3, INF)),  # neg * neg, l_x = -inf
        ((1, INF, -4, -3), (-INF, -3)),  # pos * neg, u_x = inf
        ((-INF, -1, 3, 4), (-INF, -3)),  # neg * pos, l_x = -inf
        ((-INF, 2, 3, 4), (-INF, 8)),  # 0 in x, y pos, l_x = -inf
        ((-1, INF, 3, 4), (-4, INF)),  # 0 in x, y pos, u_x = inf
        ((-1, INF, -4, -3), (-INF, 4)),  # 0 in x, y neg, u_x = inf
        ((-INF, 2, -4, -3), (-8, INF)),  # 0 in x, y neg, l_x = -inf
        ((1, INF, -3, 4), (-INF, INF)),  # x pos, 0 in y, u_x = inf
        ((-INF, -1, -3, 4), (-INF, INF)),  # x neg, 0 in y, l_x = -inf
    ],
)
def test_multiply_infinite_sign_cases(args, expected) -> None:
    assert eval_multiply_bound(*args) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((-2, -1, -INF, -2), (0, 1.0)),  # neg / neg, l_y = -inf
        ((-INF, -1, -4, -2), (0.25, INF)),  # neg / neg, l_x = -inf
        ((1, INF, -4, -2), (-INF, -0.25)),  # pos / neg, u_x = inf
        ((1, 2, -INF, -2), (-0.5, 0)),  # pos / neg, l_y = -inf
        ((-INF, -1, 2, 4), (-INF, -0.25)),  # neg / pos, l_x = -inf
        ((-2, -1, 2, INF), (-1.0, 0)),  # neg / pos, u_y = inf
        ((-INF, 2, 2, 4), (-INF, 1.0)),  # 0 in x, y pos, l_x = -inf
        ((-1, INF, 2, 4), (-0.5, INF)),  # 0 in x, y pos, u_x = inf
        ((-1, INF, -4, -2), (-INF, 0.5)),  # 0 in x, y neg, u_x = inf
        ((-INF, 2, -4, -2), (-1.0, INF)),  # 0 in x, y neg, l_x = -inf
    ],
)
def test_div_infinite_sign_cases(args, expected) -> None:
    lo, hi = eval_div_bound(*args)
    assert lo == pytest.approx(expected[0])
    assert hi == pytest.approx(expected[1])
