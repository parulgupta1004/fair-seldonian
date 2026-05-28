import numpy as np
import torch

from fair_seldonian.config import SeldonianConfig
from fair_seldonian.constraints.inequalities import Inequality
from fair_seldonian.data.synthetic import data_split, get_data
from fair_seldonian.models.logistic_regression import (
    eval_ghat,
    fHat,
    ghat,
    predict,
    simple_logistic,
)


def _split(n=500):
    data = get_data(
        N=n, features=5, t_ratio=0.5, tp0_ratio=0.4, tp1_ratio=0.6, random_seed=42
    )
    return data_split(frac=0.5, All=data, random_state=1, mTest=0.2)


def test_predict_range():
    p = predict(
        torch.randn(5, dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        np.random.randn(20, 5),
    )
    assert p.shape[0] == 20 and (p >= 0).all() and (p <= 1).all()


def test_predict_none():
    assert (predict(None, None, np.zeros((5, 5))) == 1.0).all()


def test_predict_zero_weights():
    p = predict(
        torch.zeros(5, dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        np.random.randn(10, 5),
    )
    assert torch.allclose(p, torch.full_like(p, 0.5))


def test_fhat():
    loss = fHat(
        torch.randn(5, dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        np.random.randn(30, 5),
        np.random.randint(0, 2, 30).astype(float),
    )
    assert loss.dim() == 0 and float(loss) <= 0


def test_simple_logistic():
    Xt, Yt, _, _, _, _ = _split()
    theta, theta1 = simple_logistic(Xt, Yt)
    assert theta.shape[0] == 5 and theta1.shape[0] == 1


def test_eval_ghat_all_modes():
    Xt, Yt, Tt, Xe, Ye, Te = _split()
    theta, theta1 = simple_logistic(Xt, Yt)
    for mode in ["base", "mod", "bound", "const", "opt"]:
        assert eval_ghat(theta, theta1, Xe, Ye, Te, mode) is not None


def test_ghat_all_modes():
    Xt, Yt, Tt, _, _, _ = _split()
    theta, theta1 = simple_logistic(Xt, Yt)
    for mode in ["base", "mod", "bound", "const", "opt"]:
        assert ghat(theta, theta1, Xt, Yt, Tt, 0.4, mode) is not None


def test_eval_ghat_custom_config():
    config = SeldonianConfig(delta=0.01)
    Xt, Yt, Tt, Xe, Ye, Te = _split()
    theta, theta1 = simple_logistic(Xt, Yt)
    for mode in ["base", "mod", "bound", "const", "opt"]:
        assert eval_ghat(theta, theta1, Xe, Ye, Te, mode, config) is not None


def test_ghat_ttest_config():
    config = SeldonianConfig(inequality=Inequality.T_TEST)
    Xt, Yt, Tt, _, _, _ = _split()
    theta, theta1 = simple_logistic(Xt, Yt)
    assert ghat(theta, theta1, Xt, Yt, Tt, 0.4, "base", config) is not None


def test_smaller_delta_widens_bound():
    Xt, Yt, Tt, _, _, _ = _split()
    theta, theta1 = simple_logistic(Xt, Yt)
    loose = float(
        eval_ghat(theta, theta1, Xt, Yt, Tt, "base", SeldonianConfig(delta=0.10))
    )
    tight = float(
        eval_ghat(theta, theta1, Xt, Yt, Tt, "base", SeldonianConfig(delta=0.01))
    )
    assert tight >= loose
