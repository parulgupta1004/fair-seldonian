from dataclasses import dataclass

from .constraints.inequalities import Inequality


@dataclass(frozen=True)
class SeldonianConfig:
    """Configuration for the Seldonian algorithm."""

    delta: float = 0.05
    inequality: Inequality = Inequality.HOEFFDING_INEQUALITY
    constraint: str = "TP(1) TP(0) - abs 0.25 TP(1) * -"
    candidate_ratio: float = 0.40

    def __post_init__(self):
        if not 0 < self.delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if not 0 < self.candidate_ratio < 1:
            raise ValueError(
                f"candidate_ratio must be in (0, 1), got {self.candidate_ratio}"
            )


DEFAULT_CONFIG = SeldonianConfig()
