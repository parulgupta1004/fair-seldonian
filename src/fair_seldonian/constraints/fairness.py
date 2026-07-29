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
  ``FP(g) / (FP(g) + TN(g))``;
* error (misclassification) rate  ``P(Y-hat != Y | A = g)`` is ``FP(g) + FN(g)``.

Every constraint encodes ``g(theta) <= 0`` and is bounded by a tolerance
``epsilon`` (smaller is stricter). Most bound a between-group *gap* and take a
``groups`` pair; :func:`error_rate` instead bounds a *single* group's error rate
and takes one ``group``. Group labels are matched against ``str(T)``, so the
defaults ``("1", "0")`` line up with a 0/1 sensitive column.

.. note::

   ``equal_opportunity`` and ``equalized_odds`` divide by per-group base rates to
   form conditional rates. Confidence bounds on a ratio are looser than on a
   difference, so these need more data (or a tighter ``inequality`` such as
   ``Inequality.T_TEST``) to certify than the division-free constraints
   (``demographic_parity``, ``error_rate``, ``error_rate_parity``).
"""

from __future__ import annotations

__all__ = [
    "FAIRNESS_CONSTRAINTS",
    "demographic_parity",
    "equal_opportunity",
    "equalized_odds",
    "error_rate",
    "error_rate_parity",
]


def _check_epsilon(epsilon: float) -> None:
    """Reject a negative tolerance."""
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")


def _check_label(group: object) -> str:
    """Validate a single group label and return it as a postfix-safe string."""
    g = str(group)
    if not g:
        raise ValueError("group labels must be non-empty")
    if any(c.isspace() for c in g) or "(" in g or ")" in g:
        raise ValueError(
            f"group label {g!r} must not contain whitespace or parentheses"
        )
    return g


def _validate(epsilon: float, groups: tuple[object, object]) -> tuple[str, str]:
    """Validate the tolerance and a pair of group labels; return them as strings."""
    _check_epsilon(epsilon)
    if len(groups) != 2:
        raise ValueError(f"groups must contain exactly two labels, got {groups!r}")
    g1, g0 = (_check_label(g) for g in groups)
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


def _error_rate(g: str) -> str:
    """Misclassification rate for group ``g``:  ``FP(g) + FN(g)``."""
    return f"FP({g}) FN({g}) +"


def demographic_parity(
    epsilon: float = 0.1, groups: tuple[object, object] = ("1", "0")
) -> str:
    """Demographic parity: equal predicted-positive rate across groups.

    Bounds ``|P(Y-hat = 1 | A = g1) - P(Y-hat = 1 | A = g0)| <= epsilon``.

    Also called *statistical parity* or the *independence* criterion: it requires
    the prediction to be statistically independent of the sensitive attribute, so
    each group is predicted positive at the same rate **regardless of the true
    label** ``Y``. Because it ignores ``Y``, it is a natural target for
    *allocative* decisions - where a positive prediction grants access to a
    benefit and the goal is equal access across groups - but it can be satisfied
    by a model that is deliberately less accurate for one group. It is
    division-free and therefore the easiest of these criteria to certify.

    References:
        - Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012).
          Fairness through awareness. *ITCS '12*. https://arxiv.org/abs/1104.3913
        - Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine
          Learning: Limitations and Opportunities* (independence criterion).
          MIT Press. https://fairmlbook.org

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
    ``TPR(g) = P(Y-hat = 1 | Y = 1, A = g)``.

    Introduced by Hardt, Price & Srebro (2016) as the single-error-rate relaxation
    of equalized odds. Unlike demographic parity it is *conditioned on the true
    label*, so it only asks that qualified members (those with ``Y = 1``) have an
    equal chance of a positive prediction across groups, and does not penalise a
    model for differing base rates between groups. It is the right target when the
    cost of a missed positive (a false negative) is what must be shared fairly -
    e.g. equal recall among applicants who truly qualify.

    References:
        - Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in
          supervised learning. *NeurIPS 2016*. https://arxiv.org/abs/1610.02413

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

    where ``FPR(g) = P(Y-hat = 1 | Y = 0, A = g)``.

    Introduced by Hardt, Price & Srebro (2016) as the *separation* criterion: the
    prediction must be independent of the sensitive attribute *conditional on the
    true label*, i.e. groups must be matched on both true-positive **and**
    false-positive rate. It therefore controls unfairness for both qualified and
    unqualified members, and is stricter than :func:`equal_opportunity`, which
    bounds the true-positive gap alone. Here both gaps are required to fit within a
    single tolerance ``epsilon``.

    References:
        - Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in
          supervised learning. *NeurIPS 2016*. https://arxiv.org/abs/1610.02413

    :param epsilon: maximum allowed sum of the true- and false-positive gaps.
    :param groups: the two sensitive-attribute values ``(g1, g0)`` to compare.
    :return: the constraint in postfix notation.
    """
    g1, g0 = _validate(epsilon, groups)
    tpr_gap = f"{_true_positive_rate(g1)} {_true_positive_rate(g0)} - abs"
    fpr_gap = f"{_false_positive_rate(g1)} {_false_positive_rate(g0)} - abs"
    return f"{tpr_gap} {fpr_gap} + {_num(epsilon)} -"


def error_rate(epsilon: float = 0.1, group: object = "1") -> str:
    """Error rate: bound a single group's misclassification rate.

    Bounds ``P(Y-hat != Y | A = group) = FP(group) + FN(group) <= epsilon`` - one
    minus that group's accuracy. Unlike the other builders this is a *performance*
    (behavioral) bound on one group rather than a between-group comparison: use it
    to cap how often the model errs on a chosen subpopulation (or on the whole
    sample, when the data is treated as a single group). Pair it with a parity
    constraint when you want error to be both *low* and *equal*. It is
    division-free.

    References:
        - Thomas, P. S., da Silva, B. C., Barto, A. G., Giguère, S., Brun, Y., &
          Brunskill, E. (2019). Preventing undesirable behavior of intelligent
          machines. *Science*, 366(6468), 999-1004 (behavioral constraints such
          as bounded error). https://doi.org/10.1126/science.aag3311

    :param epsilon: maximum allowed misclassification rate for the group.
    :param group: the sensitive-attribute value whose error rate is bounded.
    :return: the constraint in postfix notation.
    """
    _check_epsilon(epsilon)
    g = _check_label(group)
    return f"{_error_rate(g)} {_num(epsilon)} -"


def error_rate_parity(
    epsilon: float = 0.1, groups: tuple[object, object] = ("1", "0")
) -> str:
    """Error-rate parity: equal misclassification rate across groups.

    Bounds ``|err(g1) - err(g0)| <= epsilon``, where
    ``err(g) = P(Y-hat != Y | A = g) = FP(g) + FN(g)``.

    Also called *overall accuracy equality*: it asks that the model be right
    equally often for each group, without constraining *which* kind of error
    (false positive or false negative) may differ. It is division-free like
    demographic parity. Note that equal *total* error can still hide a group whose
    mistakes are mostly false negatives while another's are mostly false positives;
    use :func:`equalized_odds` when the error *types* must match too.

    References:
        - Berk, R., Heidari, H., Jabbari, S., Kearns, M., & Roth, A. (2021).
          Fairness in criminal justice risk assessments: The state of the art.
          *Sociological Methods & Research*, 50(1), 3-44 (overall accuracy
          equality). https://arxiv.org/abs/1703.09207
        - Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine
          Learning: Limitations and Opportunities*. MIT Press. https://fairmlbook.org

    :param epsilon: maximum allowed gap between the groups' error rates.
    :param groups: the two sensitive-attribute values ``(g1, g0)`` to compare.
    :return: the constraint in postfix notation.
    """
    g1, g0 = _validate(epsilon, groups)
    return f"{_error_rate(g1)} {_error_rate(g0)} - abs {_num(epsilon)} -"


#: Group-parity builders, keyed by name. Each takes ``(epsilon, groups)`` and
#: bounds a between-group gap. :func:`error_rate` is a single-group performance
#: bound with a different signature and is intentionally not included here.
FAIRNESS_CONSTRAINTS = {
    "demographic_parity": demographic_parity,
    "equal_opportunity": equal_opportunity,
    "equalized_odds": equalized_odds,
    "error_rate_parity": error_rate_parity,
}
