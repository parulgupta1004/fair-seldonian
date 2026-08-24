import numpy as np
import pytest

from fair_seldonian.algorithms.qsa import QSA, safety_test, split_candidate_safety
from fair_seldonian.config import DEFAULT_CONFIG, SeldonianConfig
from fair_seldonian.constraints.expression_tree import constraint_groups
from fair_seldonian.constraints.inequalities import Inequality
from fair_seldonian.data.synthetic import data_split, get_data
from fair_seldonian.models.logistic_regression import simple_logistic


def _split(
    n: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = get_data(
        N=n, features=5, t_ratio=0.5, tp0_ratio=0.4, tp1_ratio=0.6, random_seed=42
    )
    return data_split(frac=0.5, all_data=data, random_state=1, m_test=0.2)


def test_qsa() -> None:
    Xt, Yt, Tt, _, _, _ = _split()
    theta, theta1, passed, _ = QSA(Xt, Yt, Tt, "base", None, None)
    assert theta.shape[0] == 5 and theta1.shape[0] == 1 and isinstance(passed, bool)


def test_qsa_returns_tuple() -> None:
    Xt, Yt, Tt, _, _, _ = _split()
    result = QSA(Xt, Yt, Tt, "base", None, None)
    assert isinstance(result, tuple) and len(result) == 4


def test_qsa_opt() -> None:
    Xt, Yt, Tt, _, _, _ = _split()
    theta, theta1, _, _ = QSA(Xt, Yt, Tt, "opt", None, None)
    assert theta is not None and theta1 is not None


def test_safety_test_all_modes() -> None:
    Xt, Yt, Tt, _, _, _ = _split(n=500)
    theta, theta1 = simple_logistic(Xt, Yt)
    for mode in ["base", "mod", "bound", "const", "opt"]:
        assert isinstance(safety_test(theta, theta1, Xt, Yt, Tt, mode), bool)


def test_qsa_custom_config() -> None:
    config = SeldonianConfig(delta=0.01, candidate_ratio=0.5)
    Xt, Yt, Tt, _, _, _ = _split()
    theta, theta1, passed, _ = QSA(Xt, Yt, Tt, "base", None, None, config)
    assert theta is not None and theta1 is not None and isinstance(passed, bool)


def test_qsa_ttest_config() -> None:
    config = SeldonianConfig(inequality=Inequality.T_TEST)
    Xt, Yt, Tt, _, _, _ = _split()
    theta, theta1, passed, _ = QSA(Xt, Yt, Tt, "base", None, None, config)
    assert isinstance(passed, bool)


def test_safety_test_custom_config() -> None:
    config = SeldonianConfig(delta=0.10)
    Xt, Yt, Tt, _, _, _ = _split(n=500)
    theta, theta1 = simple_logistic(Xt, Yt)
    assert isinstance(safety_test(theta, theta1, Xt, Yt, Tt, "base", config), bool)


@pytest.mark.parametrize("mode", ["base", "opt", "affine"])
def test_safety_test_agrees_with_the_verdict_qsa_reports(mode: str) -> None:
    """``safety_test`` and ``QSA`` must apply the same pass criterion.

    ``QSA`` does not call ``safety_test``; it evaluates the bound itself. So the
    rule lived in two places and could drift, leaving a model that
    ``safety_test`` certifies but ``QSA`` reports as rejected.
    """
    Xt, Yt, Tt, _, _, _ = _split()
    result = QSA(Xt, Yt, Tt, mode, None, None)
    cand_X, safe_X, cand_Y, safe_Y, cand_T, safe_T = split_candidate_safety(
        Xt, Yt, Tt, DEFAULT_CONFIG.candidate_ratio
    )
    direct = safety_test(result.theta, result.theta1, safe_X, safe_Y, safe_T, mode)
    assert direct == result.passed_safety
    assert (result.diagnostics.failure_mode == "solution_found") == result.passed_safety


def test_qsa_rejects_a_group_the_data_does_not_contain() -> None:
    """An upcast ``T`` used to look like an ordinary "no solution found".

    Group labels are matched as strings, so a float column makes every mask empty
    and the bound fails closed to ``+inf``. That is a type error, not a fairness
    result, and it should not be reported as one.
    """
    Xt, Yt, Tt, _, _, _ = _split(n=400)
    with pytest.raises(ValueError, match="appear in no row of T"):
        QSA(Xt, Yt, np.asarray(Tt, dtype=float), "base", None, None)


def test_constraint_groups_reads_every_group_the_constraint_names() -> None:
    assert constraint_groups("TP(1) TP(0) - abs 0.1 -") == ["0", "1"]
    assert constraint_groups("TPR(F) TPR(M) - abs 0.05 -") == ["F", "M"]
    # Constants and operators contribute no groups, and repeats collapse.
    assert constraint_groups("PR(1) PR(1) - abs 0.2 -") == ["1"]
