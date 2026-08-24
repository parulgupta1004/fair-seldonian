from __future__ import annotations

from dataclasses import dataclass

from .constraints.expression_tree import validate_constraint
from .constraints.inequalities import Inequality


@dataclass(frozen=True)
class SeldonianConfig:
    """Configuration for the Seldonian algorithm.

    The ``constraint`` is the fairness/behavioral requirement that
    :func:`~fair_seldonian.algorithms.qsa.QSA` must certify, given as a
    reverse-Polish (postfix) string over the per-group confusion-matrix cells
    ``TP(g)``, ``FP(g)``, ``FN(g)``, ``TN(g)``. You can supply it two ways:

    * a **built-in fairness constraint** from
      :mod:`fair_seldonian.constraints.fairness` (recommended) -
      :func:`~fair_seldonian.constraints.fairness.demographic_parity`,
      :func:`~fair_seldonian.constraints.fairness.equal_opportunity`,
      :func:`~fair_seldonian.constraints.fairness.equalized_odds`,
      :func:`~fair_seldonian.constraints.fairness.error_rate`, or
      :func:`~fair_seldonian.constraints.fairness.error_rate_parity`; or
    * a **custom postfix string** you write yourself.

    Both produce the same kind of string, so they are interchangeable::

        from fair_seldonian import SeldonianConfig, demographic_parity

        SeldonianConfig(constraint=demographic_parity(epsilon=0.1))
        SeldonianConfig(constraint="PR(1) PR(0) - abs 0.1 -")

    The constraint is validated on construction (via
    :func:`~fair_seldonian.constraints.expression_tree.validate_constraint`), so a
    malformed custom string raises ``ValueError`` immediately rather than failing
    inside QSA.

    :param delta: the constraint must hold with probability >= 1 - delta.
    :param inequality: concentration inequality used for the confidence bound.
    :param constraint: the postfix constraint string (see above).
    :param candidate_ratio: fraction of data used to pick the candidate solution.
    :param optimizer: any ``method`` accepted by :func:`scipy.optimize.minimize`.
    :param max_iter: iteration cap handed to the optimizer.
    :param penalty: weight on the constraint violation in candidate selection.
        Candidate selection minimises ``log_loss + penalty * max(0, u)`` where ``u``
        is the predicted upper bound on the constraint.

        A continuous penalty matters more than it looks. The tempting alternative
        is a hard barrier - return some large constant plus ``u`` when infeasible
        and the loss when feasible - but SciPy's Powell convergence test is
        *relative*: with an objective of order 1e4 and the default ``ftol=1e-4``
        the stopping threshold is about 1.0, while ``u`` varies by only about 1e-2
        over the whole parameter space. Powell then reports success after a single
        iteration while still infeasible, and ``max_iter`` never binds. An exact
        penalty keeps the objective order 1 and gives the optimizer usable signal
        on the infeasible side.
    """

    delta: float = 0.05
    inequality: Inequality = Inequality.HOEFFDING_INEQUALITY
    constraint: str = "TP(1) TP(0) - abs 0.25 TP(1) * -"
    candidate_ratio: float = 0.40
    optimizer: str = "Powell"
    max_iter: int = 10000
    penalty: float = 100.0

    def __post_init__(self) -> None:
        if not 0 < self.delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if not 0 < self.candidate_ratio < 1:
            raise ValueError(
                f"candidate_ratio must be in (0, 1), got {self.candidate_ratio}"
            )
        if self.penalty <= 0:
            raise ValueError(f"penalty must be positive, got {self.penalty}")
        validate_constraint(self.constraint)


DEFAULT_CONFIG = SeldonianConfig()
