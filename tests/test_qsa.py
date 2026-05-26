from fair_seldonian.algorithms.qsa import QSA, safety_test
from fair_seldonian.models.logistic_regression import simple_logistic
from fair_seldonian.data.synthetic import get_data, data_split


def _split(n=1000):
    data = get_data(N=n, features=5, t_ratio=0.5, tp0_ratio=0.4, tp1_ratio=0.6, random_seed=42)
    return data_split(frac=0.5, All=data, random_state=1, mTest=0.2)


def test_qsa():
    Xt, Yt, Tt, _, _, _ = _split()
    theta, theta1, passed = QSA(Xt, Yt, Tt, "base", None, None)
    assert theta.shape[0] == 5 and theta1.shape[0] == 1 and isinstance(passed, bool)

def test_qsa_opt():
    Xt, Yt, Tt, _, _, _ = _split()
    theta, theta1, _ = QSA(Xt, Yt, Tt, "opt", None, None)
    assert theta is not None and theta1 is not None

def test_safety_test_all_modes():
    Xt, Yt, Tt, _, _, _ = _split(n=500)
    theta, theta1 = simple_logistic(Xt, Yt)
    for mode in ["base", "mod", "bound", "const", "opt"]:
        assert isinstance(safety_test(theta, theta1, Xt, Yt, Tt, mode), bool)
