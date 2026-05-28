from dataclasses import dataclass

from .constraints.inequalities import Inequality


@dataclass(frozen=True)
class SeldonianConfig:
    """Configuration for the Seldonian algorithm."""

    delta: float = 0.05
    inequality: Inequality = Inequality.HOEFFDING_INEQUALITY
    constraint: str = "TP(1) TP(0) - abs 0.25 TP(1) * -"
    candidate_ratio: float = 0.40


DEFAULT_CONFIG = SeldonianConfig()
