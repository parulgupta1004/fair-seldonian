from dataclasses import FrozenInstanceError, replace

import pytest

from fair_seldonian.config import DEFAULT_CONFIG, SeldonianConfig
from fair_seldonian.constraints.inequalities import Inequality


def test_defaults() -> None:
    config = SeldonianConfig()
    assert config.delta == 0.05
    assert config.inequality == Inequality.HOEFFDING_INEQUALITY
    assert config.constraint == "TP(1) TP(0) - abs 0.25 TP(1) * -"
    assert config.candidate_ratio == 0.40


def test_default_config_matches_defaults() -> None:
    assert DEFAULT_CONFIG == SeldonianConfig()


def test_custom_values() -> None:
    config = SeldonianConfig(
        delta=0.01,
        inequality=Inequality.T_TEST,
        constraint="FP(1) FP(0) -",
        candidate_ratio=0.6,
    )
    assert config.delta == 0.01
    assert config.inequality == Inequality.T_TEST
    assert config.constraint == "FP(1) FP(0) -"
    assert config.candidate_ratio == 0.6


def test_frozen() -> None:
    config = SeldonianConfig()
    with pytest.raises(FrozenInstanceError):
        config.delta = 0.1  # pyrefly: ignore[read-only]


def test_replace() -> None:
    config = replace(DEFAULT_CONFIG, delta=0.01)
    assert config.delta == 0.01
    assert config.candidate_ratio == DEFAULT_CONFIG.candidate_ratio


@pytest.mark.parametrize("delta", [0, 1, -0.1, 1.5])
def test_invalid_delta_raises(delta: float) -> None:
    with pytest.raises(ValueError):
        SeldonianConfig(delta=delta)


@pytest.mark.parametrize("ratio", [0, 1, -0.5, 2.0])
def test_invalid_candidate_ratio_raises(ratio: float) -> None:
    with pytest.raises(ValueError):
        SeldonianConfig(candidate_ratio=ratio)
