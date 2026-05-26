import math

from fair_seldonian.constraints.bounds import (
    eval_abs_bound,
    eval_add_bound,
    eval_div_bound,
    eval_math_bound,
    eval_multiply_bound,
    eval_subtract_bound,
)


def test_add():
    assert eval_add_bound(1, 3, 2, 4) == (3, 7)


def test_subtract():
    assert eval_subtract_bound(1, 3, 2, 4) == (-3, 1)


def test_multiply():
    assert eval_multiply_bound(2, 3, 4, 5) == (8, 15)


def test_divide():
    lo, u = eval_div_bound(2, 6, 1, 3)
    assert abs(lo - 2 / 3) < 1e-10 and abs(u - 6) < 1e-10


def test_abs_positive():
    assert eval_abs_bound(2, 5) == (2, 5)


def test_abs_negative():
    assert eval_abs_bound(-5, -2) == (2, 5)


def test_abs_straddles_zero():
    assert eval_abs_bound(-7, 3) == (0, 7)


def test_inf():
    assert eval_add_bound(-math.inf, 3, 1, 2) == (-math.inf, 5)
    assert eval_div_bound(1, 2, -1, 1) == (-math.inf, math.inf)


def test_none():
    assert eval_add_bound(None, 1, 2, 3) == (None, None)


def test_dispatch():
    assert eval_math_bound(1, 3, 2, 4, "+") == (3, 7)
    assert eval_math_bound(-3, 5, None, None, "abs") == (0, 5)
    assert eval_math_bound(1, 2, 3, 4, "%") == (None, None)
