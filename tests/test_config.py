from dataclasses import FrozenInstanceError, replace

import pytest

from fair_seldonian.config import DEFAULT_CONFIG, SeldonianConfig
from fair_seldonian.constraints.inequalities import Inequality


def test_defaults():
    config = SeldonianConfig()
    assert config.delta == 0.05
    assert config.inequality == Inequality.HOEFFDING_INEQUALITY
    assert config.constraint == "TP(1) TP(0) - abs 0.25 TP(1) * -"
    assert config.candidate_ratio == 0.40


def test_default_config_matches_defaults():
    assert DEFAULT_CONFIG == SeldonianConfig()


def test_custom_values():
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


def test_frozen():
    config = SeldonianConfig()
    with pytest.raises(FrozenInstanceError):
        config.delta = 0.1


def test_replace():
    config = replace(DEFAULT_CONFIG, delta=0.01)
    assert config.delta == 0.01
    assert config.candidate_ratio == DEFAULT_CONFIG.candidate_ratio
