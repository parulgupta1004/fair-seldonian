import pandas as pd

from fair_seldonian.data.synthetic import data_split, get_data


def test_get_data():
    data = get_data(
        N=100, features=3, t_ratio=0.5, tp0_ratio=0.4, tp1_ratio=0.6, random_seed=42
    )
    assert isinstance(data, pd.DataFrame) and len(data) == 100


def test_data_split():
    data = get_data(
        N=200, features=5, t_ratio=0.5, tp0_ratio=0.4, tp1_ratio=0.6, random_seed=42
    )
    Xe, Ye, Te, Xt, Yt, Tt = data_split(frac=0.5, All=data, random_state=1, mTest=0.2)
    assert len(Xe) > 0 and len(Xt) > 0
    assert len(Ye) == len(Xe) and len(Yt) == len(Xt)
