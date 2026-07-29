"""Ready-to-use fairness constraints.

Each function returns a constraint in the reverse-Polish (postfix) notation that
:class:`~fair_seldonian.config.SeldonianConfig` expects, so you can plug a named
fairness definition straight into the algorithm without hand-writing the string::

    from fair_seldonian import SeldonianConfig, demographic_parity

    config = SeldonianConfig(constraint=demographic_parity(epsilon=0.1))

The constraints are written over the per-group confusion-matrix cells that this
library exposes as primitives - ``TP(g)``, ``FP(g)``, ``FN(g)``, ``TN(g)``. Each
is a *fraction of group* ``g`` (the four cells sum to 1 within a group), so:

* predicted-positive rate  ``P(Y-hat = 1 | A = g)`` is ``TP(g) + FP(g)``;
* true-positive rate (recall)  ``P(Y-hat = 1 | Y = 1, A = g)`` is
  ``TP(g) / (TP(g) + FN(g))``;
* false-positive rate  ``P(Y-hat = 1 | Y = 0, A = g)`` is
  ``FP(g) / (FP(g) + TN(g))``.

Every constraint encodes ``g(theta) <= 0`` and bounds a between-group gap by a
tolerance ``epsilon``; a smaller ``epsilon`` is stricter. The ``groups`` argument
gives the two sensitive-attribute values to compare, matched against ``str(T)``
(so the defaults ``("1", "0")`` line up with a 0/1 sensitive column).

.. note::

   ``equal_opportunity`` and ``equalized_odds`` divide by per-group base rates to
   form conditional rates. Confidence bounds on a ratio are looser than on a
   difference, so these need more data (or a tighter ``inequality`` such as
   ``Inequality.T_TEST``) to certify than ``demographic_parity``, which is
   division-free.
"""

from __future__ import annotations

__all__ = [
    "FAIRNESS_CONSTRAINTS",
    "demographic_parity",
    "equal_opportunity",
    "equalized_odds",
]


def _validate(epsilon: float, groups: tuple[object, object]) -> tuple[str, str]:
    """Validate the tolerance and group labels; return the labels as strings."""
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if len(groups) != 2:
        raise ValueError(f"groups must contain exactly two labels, got {groups!r}")
    g1, g0 = (str(g) for g in groups)
    for g in (g1, g0):
        if not g:
            raise ValueError("group labels must be non-empty")
        if any(c.isspace() for c in g) or "(" in g or ")" in g:
            raise ValueError(
                f"group label {g!r} must not contain whitespace or parentheses"
            )
    if g1 == g0:
        raise ValueError(f"the two groups must be distinct, got {groups!r}")
    return g1, g0


def _num(epsilon: float) -> str:
    """Render the tolerance as a single postfix token (no spaces)."""
    return repr(float(epsilon))


def _positive_rate(g: str) -> str:
    """Predicted-positive rate for group ``g``:  ``TP(g) + FP(g)``."""
    return f"TP({g}) FP({g}) +"


def _true_positive_rate(g: str) -> str:
    """True-positive rate (recall) for group ``g``:  ``TP(g) / (TP(g) + FN(g))``."""
    return f"TP({g}) TP({g}) FN({g}) + /"


def _false_positive_rate(g: str) -> str:
    """False-positive rate for group ``g``:  ``FP(g) / (FP(g) + TN(g))``."""
    return f"FP({g}) FP({g}) TN({g}) + /"


def demographic_parity(
    epsilon: float = 0.1, groups: tuple[object, object] = ("1", "0")
) -> str:
    """Demographic parity: equal predicted-positive rate across groups.

    Bounds ``|P(Y-hat = 1 | A = g1) - P(Y-hat = 1 | A = g0)| <= epsilon``. This is
    the *independence* criterion; it ignores the label ``Y`` and is division-free.

    :param epsilon: maximum allowed gap between the groups' positive rates.
    :param groups: the two sensitive-attribute values ``(g1, g0)`` to compare.
    :return: the constraint in postfix notation.
    """
    g1, g0 = _validate(epsilon, groups)
    return f"{_positive_rate(g1)} {_positive_rate(g0)} - abs {_num(epsilon)} -"


def equal_opportunity(
    epsilon: float = 0.1, groups: tuple[object, object] = ("1", "0")
) -> str:
    """Equal opportunity: equal true-positive rate (recall) across groups.

    Bounds ``|TPR(g1) - TPR(g0)| <= epsilon``, where
    ``TPR(g) = P(Y-hat = 1 | Y = 1, A = g)``. Unlike demographic parity this is
    conditioned on the true label, so it only constrains the actually-positive
    subpopulation.

    :param epsilon: maximum allowed gap between the groups' true-positive rates.
    :param groups: the two sensitive-attribute values ``(g1, g0)`` to compare.
    :return: the constraint in postfix notation.
    """
    g1, g0 = _validate(epsilon, groups)
    return (
        f"{_true_positive_rate(g1)} {_true_positive_rate(g0)} - abs {_num(epsilon)} -"
    )


def equalized_odds(
    epsilon: float = 0.1, groups: tuple[object, object] = ("1", "0")
) -> str:
    """Equalized odds: equal true- and false-positive rates across groups.

    Bounds the *sum* of the two gaps by ``epsilon``::

        |TPR(g1) - TPR(g0)| + |FPR(g1) - FPR(g0)| <= epsilon

    where ``FPR(g) = P(Y-hat = 1 | Y = 0, A = g)``. Because both gaps must fit
    within a single tolerance, equalized odds is stricter than
    :func:`equal_opportunity` (which bounds the true-positive gap alone).

    :param epsilon: maximum allowed sum of the true- and false-positive gaps.
    :param groups: the two sensitive-attribute values ``(g1, g0)`` to compare.
    :return: the constraint in postfix notation.
    """
    g1, g0 = _validate(epsilon, groups)
    tpr_gap = f"{_true_positive_rate(g1)} {_true_positive_rate(g0)} - abs"
    fpr_gap = f"{_false_positive_rate(g1)} {_false_positive_rate(g0)} - abs"
    return f"{tpr_gap} {fpr_gap} + {_num(epsilon)} -"


#: Mapping from constraint name to its builder, for discovery and iteration.
FAIRNESS_CONSTRAINTS = {
    "demographic_parity": demographic_parity,
    "equal_opportunity": equal_opportunity,
    "equalized_odds": equalized_odds,
}
