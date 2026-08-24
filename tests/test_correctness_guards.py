"""Tests for the failure modes that mostly do not announce themselves.

Nearly every test here pins down a choice whose wrong answer produces
plausible-looking numbers rather than an error: a coverage guarantee that is
quietly half what it claims, an objective that is not the one on the axis label,
a benchmark with no accuracy to trade away. Ordinary unit tests do not catch
those, because nothing raises and nothing looks absurd. The split-alignment test
is the exception - that mismatch did raise, which is why it was reachable only
at candidate ratios nobody had tried. The comments record why each assertion is what
it is, so that none of them is later "fixed" by relaxing it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import log_loss

from fair_seldonian.algorithms.qsa import cand_obj, split_candidate_safety
from fair_seldonian.config import SeldonianConfig
from fair_seldonian.constraints.affine import compile_bounds
from fair_seldonian.constraints.expression_tree import (
    ROOT_SIDES,
    child_sides,
    construct_expr_tree_base,
    eval_expr_tree_base,
)
from fair_seldonian.constraints.expression_tree_ext import construct_expr_tree
from fair_seldonian.constraints.inequalities import (
    Inequality,
    check_constraint_groups,
    eval_func_bound,
)
from fair_seldonian.data.synthetic import get_data
from fair_seldonian.models.logistic_regression import eval_ghat, f_hat, predict

PAPER_CONSTRAINT = "TP(1) TP(0) - abs 0.2 TP(1) * -"


# --------------------------------------------------------------------------
# Confidence-interval coverage
# --------------------------------------------------------------------------
def _empirical_coverage(
    n_total: int,
    n_minority: int,
    p_true: float,
    inequality: Inequality,
    trials: int = 300,
    delta: float = 0.05,
) -> float:
    rng = np.random.default_rng(7)
    T = pd.Series(np.array(["0"] * (n_total - n_minority) + ["1"] * n_minority))
    hits = 0
    for _ in range(trials):
        Y = np.zeros(n_total, dtype=int)
        Y[: n_total - n_minority] = rng.binomial(1, 0.5, n_total - n_minority)
        minority = rng.binomial(1, p_true, n_minority)
        Y[n_total - n_minority :] = minority
        pred = np.zeros(n_total)
        pred[n_total - n_minority :] = minority
        pred[: n_total - n_minority] = rng.random(n_total - n_minority)
        lo, hi = eval_func_bound(
            "TP(1)",
            pd.Series(Y),
            torch.tensor(pred),
            T,
            delta,
            inequality,
            None,
            False,
            False,
            two_sided=True,
        )
        hits += lo <= p_true <= hi
    return hits / trials


@pytest.mark.parametrize(
    "inequality", [Inequality.HOEFFDING_INEQUALITY, Inequality.EMPIRICAL_BERNSTEIN]
)
def test_interval_covers_on_a_small_minority_group(inequality: Inequality) -> None:
    """The distribution-free intervals must cover a minority group's value.

    This is the assertion the whole guarantee rests on. Building the interval from
    ``#{Y == 1}`` across *all* groups while the estimate divides by the size of one
    group makes it about 4.5x too narrow on a 500-member group inside 20,000
    samples, and drops empirical coverage to roughly 0.5 against a nominal 0.95. A
    balanced benchmark cannot detect it, because there the two counts coincide
    numerically.
    """
    assert _empirical_coverage(20_000, 500, 0.3, inequality) >= 0.95


def test_interval_width_scales_with_group_size_not_dataset_size() -> None:
    """Growing the majority around a fixed minority must not shrink its interval."""
    widths = []
    for n_total in (2_000, 20_000, 200_000):
        n_minority = 200
        T = pd.Series(np.array(["0"] * (n_total - n_minority) + ["1"] * n_minority))
        Y = pd.Series(np.ones(n_total, dtype=int))
        pred = torch.tensor(np.full(n_total, 0.5))
        lo, hi = eval_func_bound(
            "TP(1)",
            Y,
            pred,
            T,
            0.05,
            Inequality.HOEFFDING_INEQUALITY,
            None,
            False,
            False,
            two_sided=True,
        )
        widths.append(hi - lo)
    assert max(widths) - min(widths) < 1e-12


def test_group_dtype_mismatch_raises_instead_of_failing_silently() -> None:
    """Float group labels make every mask empty and every bound +inf.

    That looks exactly like a legitimate "no solution found", so the run would
    report a clean negative result rather than a type error.
    """
    check_constraint_groups(["0", "1"], pd.Series([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="upcast|appear in no row"):
        check_constraint_groups(["0", "1"], np.array([0.0, 1.0, 0.0]))


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
def test_f_hat_is_log_loss() -> None:
    """``f_hat`` must be the negative log loss, not softmax-of-probabilities.

    Passing probabilities to ``CrossEntropyLoss`` (which expects logits) yields a
    quantity floored at 0.3133 for a perfect classifier, so any curve plotted from
    it asymptotes to 0.31 rather than towards the Bayes rate, and the range is
    compressed roughly tenfold.
    """
    Y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    for probabilities in (
        np.array([0.01, 0.99, 0.99, 0.01, 0.99, 0.01, 0.99, 0.99]),
        np.array([0.2, 0.8, 0.7, 0.3, 0.9, 0.1, 0.85, 0.75]),
        np.full(8, 0.5),
    ):
        # A single feature with unit weight and no intercept makes predict() the
        # identity on the supplied probabilities' logits.
        logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
        theta = torch.tensor(np.array([1.0]))
        theta1 = torch.tensor(np.array([0.0]))
        got = -float(f_hat(theta, theta1, logits, Y))
        assert abs(got - log_loss(Y, probabilities)) < 1e-9


def test_perfect_classifier_approaches_zero_loss() -> None:
    Y = np.array([0, 1, 1, 0])
    logits = np.where(Y == 1, 30.0, -30.0).reshape(-1, 1)
    loss = -float(
        f_hat(torch.tensor(np.array([1.0])), torch.tensor(np.array([0.0])), logits, Y)
    )
    assert loss < 1e-6


# --------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------
def test_candidate_objective_stays_resolvable_against_powell_tolerance() -> None:
    """The penalty must keep the objective order 1, not order 1e4.

    SciPy's Powell convergence test is *relative*: it stops once the improvement
    falls below ``ftol * |f|``. A barrier of the form ``-10000 - u`` puts ``|f|``
    at about 1e4, so with the default ``ftol=1e-4`` the stopping threshold is
    around 1.0 while ``u`` varies by only about 1e-2 across the whole parameter
    space. Powell then declares success after a single iteration while still
    infeasible, and ``max_iter`` never binds. This asserts the property that
    prevents it: two points with materially different constraint violations must
    differ by more than Powell's relative threshold.
    """
    config = SeldonianConfig(candidate_ratio=0.6)
    data = get_data(4_000, 5, 0.5, 0.4, 0.6, random_seed=0.5)
    X = np.asarray(data.iloc[:, :-2], dtype=float)
    Y = np.asarray(data.iloc[:, -2])
    T = np.asarray(data.iloc[:, -1])

    fair = np.array([0.1, 0.05, -0.05, 0.02, 0.0, -0.1])
    unfair = np.array([3.0, 0.05, -0.05, 0.02, 2.5, -0.1])
    f_fair = cand_obj(fair, X, Y, T, "base", config)
    f_unfair = cand_obj(unfair, X, Y, T, "base", config)

    assert np.isfinite(f_fair) and np.isfinite(f_unfair)
    # Order 1, not order 1e4 - this is what makes the relative tolerance usable.
    assert max(abs(f_fair), abs(f_unfair)) < 100.0
    # And the gap must clear Powell's default relative stopping threshold.
    assert abs(f_fair - f_unfair) > 1e-4 * max(abs(f_fair), abs(f_unfair))


def test_penalty_must_be_positive() -> None:
    with pytest.raises(ValueError, match="penalty"):
        SeldonianConfig(penalty=0.0)


def test_candidate_safety_split_is_aligned() -> None:
    """X, Y and T must be split at one shared index.

    ``train_test_split(test_size=1 - r)`` rounds its test size up while
    ``np.split`` at ``int(r * n)`` rounds down, so the two disagree by a row at
    some ratios - 0.7 and 0.44 among them - leaving the group column a different
    length from the features and raising ``IndexError`` downstream.
    """
    for n in (99, 100, 101, 1_000):
        X = np.arange(n * 2, dtype=float).reshape(n, 2)
        Y = np.arange(n)
        T = np.arange(n)
        for ratio in (0.4, 0.44, 0.5, 0.6, 0.7, 0.75):
            cand_X, safe_X, cand_Y, safe_Y, cand_T, safe_T = split_candidate_safety(
                X, Y, T, ratio
            )
            # Same boundary for all three, and nothing lost or duplicated.
            assert len(cand_X) == len(cand_Y) == len(cand_T)
            assert len(safe_X) == len(safe_Y) == len(safe_T)
            assert len(cand_X) + len(safe_X) == n
            # Row i of every split still refers to the same original sample.
            assert np.array_equal(cand_X[:, 0], cand_Y * 2)
            assert np.array_equal(cand_Y, cand_T)
            assert np.array_equal(safe_Y, safe_T)


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def test_synthetic_problem_has_non_zero_bayes_error() -> None:
    """The signal feature must not determine the label.

    A construction like ``Y * uniform(0, 1)`` is zero exactly when ``Y = 0``, so
    the label is perfectly recoverable and there is no accuracy to trade away.
    """
    data = get_data(20_000, 5, 0.5, 0.4, 0.6, random_seed=0.5)
    signal = np.asarray(data.iloc[:, 0])
    Y = np.asarray(data.iloc[:, -2])
    best = max(
        np.mean((signal > threshold) == (Y == 1))
        for threshold in np.linspace(-1, 2, 61)
    )
    assert best < 0.95


def test_sensitive_feature_inclusion_is_explicit() -> None:
    with_t = get_data(2_000, 5, 0.5, 0.4, 0.6, 0.5, include_sensitive_feature=True)
    without_t = get_data(2_000, 5, 0.5, 0.4, 0.6, 0.5, include_sensitive_feature=False)
    assert np.array_equal(
        np.asarray(with_t.iloc[:, -3]), np.asarray(with_t.iloc[:, -1])
    )
    assert not np.array_equal(
        np.asarray(without_t.iloc[:, -3]), np.asarray(without_t.iloc[:, -1])
    )
    assert with_t.shape[1] == without_t.shape[1] == 7


def test_base_rate_gap_is_present() -> None:
    data = get_data(40_000, 5, 0.5, 0.4, 0.6, random_seed=0.5)
    Y = np.asarray(data.iloc[:, -2])
    T = np.asarray(data.iloc[:, -1])
    assert abs(Y[T == 0].mean() - 0.4) < 0.02
    assert abs(Y[T == 1].mean() - 0.6) < 0.02


def test_trials_with_different_seeds_are_independent_draws() -> None:
    """Distinct seeds must give distinct datasets.

    Drawing one dataset outside the trial loop and subsampling it makes the error
    bars measure subsampling noise, and at the largest size every trial sees
    identical data.
    """
    a = get_data(1_000, 5, 0.5, 0.4, 0.6, random_seed=0.1)
    b = get_data(1_000, 5, 0.5, 0.4, 0.6, random_seed=0.2)
    assert not np.array_equal(np.asarray(a.iloc[:, 0]), np.asarray(b.iloc[:, 0]))


# --------------------------------------------------------------------------
# Rate primitives
# --------------------------------------------------------------------------
def test_rate_primitive_uses_its_own_conditioning_set() -> None:
    """``TPR(g)`` is a mean over ``{T=g, Y=1}``; ``TP(g)`` over all of ``{T=g}``.

    The distinction sets the sample size an inequality may use. Conflating them
    is the same class of error as
    :func:`test_interval_covers_on_a_small_minority_group`.
    """
    from fair_seldonian.constraints.inequalities import conditioning_set, contributions

    Y = pd.Series([1, 1, 0, 0, 0])
    T = pd.Series(["g", "g", "g", "g", "g"])
    pred = torch.tensor([0.9, 0.7, 0.4, 0.2, 0.1], dtype=torch.float64)
    assert conditioning_set("TPR(g)") == ("g", 1)
    assert conditioning_set("TP(g)") == ("g", None)
    assert int(contributions("TPR(g)", Y, pred, T).numel()) == 2  # the Y=1 rows
    assert int(contributions("TP(g)", Y, pred, T).numel()) == 5  # the whole group
    assert abs(float(contributions("TPR(g)", Y, pred, T).mean()) - 0.8) < 1e-12


# --------------------------------------------------------------------------
# Betting interval
# --------------------------------------------------------------------------
@pytest.mark.parametrize("p_true", [0.5, 0.12, 0.03])
def test_betting_interval_covers(p_true: float) -> None:
    """The betting interval must deliver at least its nominal coverage.

    It is the tightest bound we ship, so it is also the one where an error would
    be least likely to show up as anything other than an over-confident result.
    """
    from fair_seldonian.constraints.inequalities import betting_interval

    rng = np.random.default_rng(11)
    hits = 0
    trials = 200
    for _ in range(trials):
        x = rng.binomial(1, p_true, 400).astype(float)
        lo, hi = betting_interval(x, 0.05, two_sided=True)
        hits += lo <= p_true <= hi
    assert hits / trials >= 0.95


def test_betting_beats_hoeffding_away_from_one_half() -> None:
    """Betting adapts to the observed spread; Hoeffding assumes the worst case."""
    from fair_seldonian.constraints.inequalities import betting_interval

    rng = np.random.default_rng(5)
    x = rng.binomial(1, 0.05, 2_000).astype(float)
    lo, hi = betting_interval(x, 0.05, two_sided=True)
    hoeffding = math.sqrt(math.log(2 / 0.05) / (2 * 2_000))
    assert (hi - lo) / 2 < 0.6 * hoeffding


# --------------------------------------------------------------------------
# Delta accounting
# --------------------------------------------------------------------------
def test_root_is_one_sided_and_abs_forces_two_sided() -> None:
    """Only the endpoints actually read should be paid for - and all of them must be."""
    tree = construct_expr_tree_base(PAPER_CONSTRAINT)
    left, right = child_sides(tree.value, ROOT_SIDES, tree.left.value, tree.right.value)
    assert right == (True, False)  # L(0.2 * TP(1)) is what subtraction reads
    abs_node = tree.left
    below_abs, _ = child_sides(abs_node.value, left, abs_node.left.value, None)
    assert below_abs == (True, True)  # abs reads both endpoints of its operand


def test_error_rate_constraint_stays_one_sided_end_to_end() -> None:
    tree = construct_expr_tree_base("FP(1) FN(1) + 0.1 -")
    sides = {}

    def walk(node, node_sides):
        if node is None:
            return
        sides[node.value] = node_sides
        ls, rs = child_sides(
            node.value,
            node_sides,
            node.left.value if node.left else None,
            node.right.value if node.right else None,
        )
        walk(node.left, ls)
        walk(node.right, rs)

    walk(tree, ROOT_SIDES)
    assert sides["FP(1)"] == (False, True) and sides["FN(1)"] == (False, True)


def test_union_bound_merge_gives_repeated_leaves_one_shared_interval() -> None:
    """Merged occurrences must agree on delta *and* sidedness.

    The merge is only sound if the repeated occurrences really are a single
    interval. Two occurrences sharing a delta but differing in sidedness would be
    two different widths, hence two failure events, doubling the true budget.
    """
    tree = construct_expr_tree(
        PAPER_CONSTRAINT, 0.05, check_bound=True, check_constant=False
    )
    seen = []

    def walk(node):
        if node is None:
            return
        if node.value == "TP(1)":
            seen.append((node.delta, node.sides))
        walk(node.left)
        walk(node.right)

    walk(tree)
    assert len(seen) > 1
    assert len(set(seen)) == 1


# --------------------------------------------------------------------------
# Affine-form bounding
# --------------------------------------------------------------------------
def test_affine_compilation_of_the_paper_constraint() -> None:
    upper, _ = compile_bounds(construct_expr_tree_base(PAPER_CONSTRAINT))
    got = {tuple(sorted(form.coefficients.items())) for form in upper}
    assert got == {
        (("TP(0)", 1.0), ("TP(1)", -1.2)),
        (("TP(0)", -1.0), ("TP(1)", 0.8)),
    }


def test_affine_bound_is_valid_and_tighter_than_tree_propagation() -> None:
    """Affine bounding must still be an upper bound, and should be much tighter."""
    config = SeldonianConfig(constraint=PAPER_CONSTRAINT, candidate_ratio=0.6)
    data = get_data(40_000, 5, 0.5, 0.4, 0.6, random_seed=0.5)
    X = np.asarray(data.iloc[:, :-2], dtype=float)
    Y = np.asarray(data.iloc[:, -2])
    T = np.asarray(data.iloc[:, -1])
    theta = torch.tensor(np.array([0.8, 0.1, -0.1, 0.05, 0.3]))
    theta1 = torch.tensor(np.array([-0.4]))

    truth = float(
        eval_expr_tree_base(
            construct_expr_tree_base(PAPER_CONSTRAINT), Y, predict(theta, theta1, X), T
        )
    )
    tree_bound = float(eval_ghat(theta, theta1, X, Y, T, "base", config))
    affine_bound = float(eval_ghat(theta, theta1, X, Y, T, "affine", config))

    assert affine_bound >= truth  # still an upper bound
    assert affine_bound < tree_bound
    # The whole point is that the gain is large, not marginal like the
    # constant-skip and union-bound tweaks (both under 2% on this constraint).
    assert (affine_bound - truth) < 0.75 * (tree_bound - truth)


def test_affine_rejects_constraints_it_cannot_represent() -> None:
    from fair_seldonian.constraints.affine import NotAffine

    # equal_opportunity divides by a per-group base rate, which is not affine.
    tree = construct_expr_tree_base("TP(1) TP(1) FN(1) + / 0.1 -")
    with pytest.raises(NotAffine):
        compile_bounds(tree)


def test_standard_fairness_definitions_are_all_affine_compilable() -> None:
    """Rate primitives bring equal opportunity and equalized odds into the fragment."""
    from fair_seldonian.constraints.fairness import (
        demographic_parity,
        equal_opportunity,
        equalized_odds,
        error_rate_parity,
    )

    expected = {
        demographic_parity: 2,
        equal_opportunity: 2,
        equalized_odds: 4,
        error_rate_parity: 2,
    }
    for builder, n_forms in expected.items():
        forms, _ = compile_bounds(construct_expr_tree_base(builder(0.1)))
        assert len(forms) == n_forms, builder.__name__


def test_overlapping_conditioning_sets_are_refused() -> None:
    """Independence is what licenses the weighted-sum bound, so overlap must raise.

    ``TP(g)`` is a mean over the whole of group ``g`` and ``TPR(g)`` over its
    ``Y=1`` rows: the sets overlap, the means are dependent, and summing their
    variances would understate the true one.
    """
    from fair_seldonian.constraints.affine import NotAffine, form_upper_bound

    forms, _ = compile_bounds(construct_expr_tree_base("TP(g) TPR(g) - abs 0.1 -"))
    Y = pd.Series([1, 0, 1, 0])
    T = pd.Series(["g"] * 4)
    pred = torch.tensor([0.6, 0.4, 0.7, 0.3], dtype=torch.float64)
    with pytest.raises(NotAffine, match="overlap"):
        for form in forms:
            form_upper_bound(form, Y, pred, T, 0.025)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def test_clopper_pearson_does_not_claim_certainty_from_zero_events() -> None:
    """0/40 violations is not evidence that the rate is below 0.05."""
    from fair_seldonian.experiments.results import clopper_pearson

    lo, hi = clopper_pearson(0, 40)
    assert lo == 0.0
    assert hi > 0.05  # cannot rule out exceeding delta on 40 trials
    assert math.isclose(hi, 0.0881, abs_tol=5e-3)
