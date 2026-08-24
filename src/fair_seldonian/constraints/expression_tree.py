from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, TypeVar, overload

from .bounds import eval_math_bound
from .inequalities import Inequality, eval_estimate, eval_func_bound

if TYPE_CHECKING:
    import torch

    from .._typing import Array, Bound

logger = logging.getLogger(__name__)


####################
# Construct Parser #
####################
class ExprTree:
    """
    An expression tree node of the constraint tree
    """

    def __init__(self, value: str) -> None:
        self.value = value
        self.left: ExprTree | None = None
        self.right: ExprTree | None = None


def is_operator(element: str) -> bool:
    return element in {"+", "-", "*", "/", "^"}


def is_mod(element: str) -> bool:
    return element == "abs"


#: Base-variable prefixes: the four confusion-matrix cells, the four
#: label-conditioned rates, and the predicted-positive/negative rates.
BASE_MEASURES = (
    "TPR",
    "FPR",
    "TNR",
    "FNR",
    "TP",
    "FP",
    "TN",
    "FN",
    "PR",
    "NR",
)


def is_func(element: str) -> bool:
    head, sep, _ = element.partition("(")
    return bool(sep) and head in BASE_MEASURES


def is_constant(t_node_value: str) -> bool:
    """Whether a token is a literal number rather than a base variable."""
    try:
        float(t_node_value)
    except (TypeError, ValueError):
        return False
    return True


# Which endpoints of a node's interval are actually consumed by its parent.
Sides = tuple[bool, bool]  # (needs_lower, needs_upper)
BOTH: Sides = (True, True)
NEITHER: Sides = (False, False)
ROOT_SIDES: Sides = (False, True)  # the safety test only ever reads U(g) vs 0


def child_sides(
    node_value: str,
    sides: Sides,
    left_value: str | None,
    right_value: str | None,
) -> tuple[Sides, Sides]:
    """Propagate "which endpoints do I need" from a node down to its children.

    A leaf only has to pay for the endpoints that are actually read. The safety
    test reads a single number - the upper bound on ``g`` - so the root needs only
    its upper endpoint, and for constraints built from ``+``, ``-`` and scaling by
    a non-negative constant that one-sidedness reaches all the way down to the
    leaves. Those leaves can then use ``ln(1/delta)`` instead of ``ln(2/delta)``.
    The half-width scales as the square root of that term, so at ``delta = 0.05``
    the interval narrows by ``1 - sqrt(ln(20)/ln(40))``, just under 10%.

    ``abs`` breaks it: ``U(|x|) = max(-L(x), U(x))`` reads both endpoints of its
    operand, so everything under an ``abs`` is two-sided. Products and quotients of
    two *variables* are treated as two-sided too, because
    :func:`~fair_seldonian.constraints.bounds.eval_multiply_bound` and
    :func:`~fair_seldonian.constraints.bounds.eval_div_bound` branch on the signs of
    both endpoints - an unguaranteed endpoint could select the wrong branch and so
    corrupt the endpoint that *is* guaranteed.

    Scaling by a constant is safe to pass through: for a fixed ``c`` every branch of
    the multiply rule collapses to ``(l*c, u*c)`` for ``c >= 0`` and ``(u*c, l*c)``
    for ``c < 0``, so only the corresponding endpoint of the variable child is read.

    :param node_value: this node's token.
    :param sides: the endpoints of *this* node that its parent needs.
    :param left_value: the left child's token, or ``None``.
    :param right_value: the right child's token, or ``None``.
    :return: ``(left_sides, right_sides)``.
    """
    need_lower, need_upper = sides
    flipped: Sides = (need_upper, need_lower)

    if is_mod(node_value):
        # abs reads both endpoints of its operand, which sits on `.left`.
        return BOTH, NEITHER
    if node_value == "+":
        return sides, sides
    if node_value == "-":
        return sides, flipped
    if node_value == "*":
        if right_value is not None and is_constant(right_value):
            return (sides if float(right_value) >= 0 else flipped), NEITHER
        if left_value is not None and is_constant(left_value):
            return NEITHER, (sides if float(left_value) >= 0 else flipped)
        return BOTH, BOTH
    # "/", "^", and anything else: be conservative.
    return BOTH, BOTH


# A well-formed group-rate token: TP/FP/TN/FN followed by a parenthesised group
# label containing no spaces or nested parentheses, e.g. ``TP(1)`` or ``FP(Male)``.
_FUNC_TOKEN_RE = re.compile(r"^(?:TPR|FPR|TNR|FNR|TP|FP|TN|FN|PR|NR)\([^()\s]+\)$")


def validate_constraint(rev_polish_notation: str) -> None:
    """Validate a reverse-Polish (postfix) constraint string.

    Checks that every token is recognized, that each operator/``abs`` has enough
    operands, and that the whole expression reduces to a single value - i.e. that
    :func:`construct_expr_tree_base` can turn it into an evaluable tree. This is
    what :class:`~fair_seldonian.config.SeldonianConfig` runs on its ``constraint``
    so that a malformed custom string fails immediately instead of deep inside the
    algorithm.

    :param rev_polish_notation: the postfix constraint string to validate.
    :raises ValueError: if the string is empty or not a valid postfix expression.
    """
    if not rev_polish_notation or not rev_polish_notation.strip():
        raise ValueError("constraint must be a non-empty postfix expression")
    stack_size = 0
    for token in rev_polish_notation.split(" "):
        if token == "":
            raise ValueError(
                f"constraint {rev_polish_notation!r} has an empty token; "
                "use single spaces between tokens"
            )
        if is_operator(token):
            if stack_size < 2:
                raise ValueError(
                    f"operator {token!r} needs two operands in {rev_polish_notation!r}"
                )
            stack_size -= 1  # pop two operands, push one result
        elif is_mod(token):
            if stack_size < 1:
                raise ValueError(f"'abs' needs one operand in {rev_polish_notation!r}")
            # pop one operand, push one result: net change is zero
        elif _FUNC_TOKEN_RE.match(token):
            stack_size += 1
        else:
            try:
                float(token)
            except ValueError:
                raise ValueError(
                    f"unrecognized token {token!r} in constraint "
                    f"{rev_polish_notation!r}; expected a number, an operator "
                    "(+ - * / ^), 'abs', or a group rate like 'TP(1)'"
                ) from None
            stack_size += 1
    if stack_size != 1:
        raise ValueError(
            f"constraint {rev_polish_notation!r} is not a valid postfix expression "
            "(it does not reduce to a single value)"
        )


_NodeT = TypeVar("_NodeT", bound=ExprTree)


@overload
def construct_expr_tree_base(
    rev_polish_notation: str, node_class: None = None
) -> ExprTree: ...
@overload
def construct_expr_tree_base(
    rev_polish_notation: str, node_class: type[_NodeT]
) -> _NodeT: ...
def construct_expr_tree_base(
    rev_polish_notation: str, node_class: type[ExprTree] | None = None
) -> ExprTree:
    """
    Returns root of constructed tree for given postfix expression

    :param rev_polish_notation: string with space as delimiter ' '
    :param node_class: the tree node class to use (default: ExprTree)
    :return: ExprTree node
    """
    if node_class is None:
        node_class = ExprTree
    tokens = rev_polish_notation.split(" ")
    stack: list[ExprTree] = []
    for element in tokens:
        if not is_operator(element) and not is_mod(element):
            t = node_class(element)
            stack.append(t)
        else:
            if is_mod(element):
                t = node_class(element)
                t1 = None
                t2 = stack.pop()
            else:
                t = node_class(element)
                t1 = stack.pop()
                t2 = stack.pop()
            t.right = t1
            t.left = t2
            stack.append(t)
    t = stack.pop()
    return t


#################
# Evaluate tree #
#################
def eval_expr_tree_base(
    t_node: ExprTree | None,
    Y: Array | None,
    predicted_Y: torch.Tensor | None,
    T: Array | None,
) -> Bound | None:
    """
    A utility function to evaluate estimate of the expression tree

    :param t_node: ExprTree node
    :param Y: pandas::Series
    :param predicted_Y: tensor
    :param T: pandas::Series
    :return: estimate value: float
    """
    if t_node is not None:
        x = eval_expr_tree_base(t_node.left, Y, predicted_Y, T)
        y = eval_expr_tree_base(t_node.right, Y, predicted_Y, T)
        if x is None:
            if is_func(t_node.value):
                # Function nodes require the dataset to compute an estimate.
                assert Y is not None and predicted_Y is not None and T is not None
                return eval_estimate(t_node.value, Y, predicted_Y, T)
            return float(t_node.value)
        elif y is None:
            if is_mod(t_node.value):
                return abs(float(x))
            return None
        else:
            if t_node.value == "+":
                return x + y
            elif t_node.value == "-":
                return x - y
            elif t_node.value == "*":
                return x * y
            elif t_node.value == "^":
                return x**y
            elif t_node.value == "/":
                return x / y
            elif is_func(t_node.value):
                # Function nodes require the dataset to compute an estimate.
                assert Y is not None and predicted_Y is not None and T is not None
                return eval_estimate(t_node.value, Y, predicted_Y, T)
            elif is_mod(t_node.value):
                return abs(float(x))
            return None
    return None


##########################
# Evaluate conf interval #
##########################
def _eval_node_bounds(
    t_node: ExprTree,
    l_x: Bound | None,
    u_x: Bound | None,
    l_y: Bound | None,
    u_y: Bound | None,
    delta: float,
    Y: Array,
    predicted_Y: torch.Tensor,
    T: Array,
    inequality: Inequality,
    candidate_safety_ratio: float | None,
    predict_bound: bool,
    modified_h: bool,
    sides: Sides = BOTH,
) -> tuple[Bound | None, Bound | None]:
    def leaf() -> tuple[Bound, Bound]:
        return eval_func_bound(
            t_node.value,
            Y,
            predicted_Y,
            T,
            delta,
            inequality,
            candidate_safety_ratio,
            predict_bound,
            modified_h,
            two_sided=sides[0] and sides[1],
        )

    if l_x is None and u_x is None:
        if is_func(t_node.value):
            return leaf()
        return float(t_node.value), float(t_node.value)
    elif l_y is None and u_y is None:
        if is_mod(t_node.value):
            return eval_math_bound(l_x, u_x, l_y, u_y, "abs")
        return None, None
    else:
        if is_operator(t_node.value):
            return eval_math_bound(l_x, u_x, l_y, u_y, t_node.value)
        elif is_func(t_node.value):
            return leaf()
        elif is_mod(t_node.value):
            return eval_math_bound(l_x, u_x, l_y, u_y, "abs")
        return None, None


def eval_expr_tree_conf_interval_base(
    t_node: ExprTree | None,
    Y: Array,
    predicted_Y: torch.Tensor,
    T: Array,
    delta: float,
    inequality: Inequality,
    candidate_safety_ratio: float | None,
    predict_bound: bool,
    modified_h: bool,
    sides: Sides = ROOT_SIDES,
) -> tuple[Bound | None, Bound | None]:
    """
    To evaluate confidence interval of the expression tree

    :param t_node: ExprTree node
    :param Y: pandas::Series The true labels of the dataset
    :param predicted_Y: tensor The predicted labels of the dataset
    :param T: pandas::Series The sensitive attributes of the dataset
    :param delta: float in [0, 1] The significance level
    :param inequality: Enum The inequality to be used - Hoeffding/T-test
    :param candidate_safety_ratio: The candidate to safety ratio used in the experiment
    :param predict_bound: Whether we are finding bound for candidate
        or safety data
    :param modified_h: Whether modified confidence bound is used
    :param sides: which endpoints of this node's interval the caller consumes.
        Defaults to the root's requirement (upper only); see :func:`child_sides`.
    :return: upper and lower bound of the estimate of the constraint
    """
    if t_node is not None:
        if t_node.right is not None and t_node.right.value is not None:
            child_delta = delta / 2
        else:
            child_delta = delta
        left_sides, right_sides = child_sides(
            t_node.value,
            sides,
            t_node.left.value if t_node.left is not None else None,
            t_node.right.value if t_node.right is not None else None,
        )
        l_x, u_x = eval_expr_tree_conf_interval_base(
            t_node.left,
            Y,
            predicted_Y,
            T,
            child_delta,
            inequality,
            candidate_safety_ratio,
            predict_bound,
            modified_h,
            left_sides,
        )
        l_y, u_y = eval_expr_tree_conf_interval_base(
            t_node.right,
            Y,
            predicted_Y,
            T,
            child_delta,
            inequality,
            candidate_safety_ratio,
            predict_bound,
            modified_h,
            right_sides,
        )
        return _eval_node_bounds(
            t_node,
            l_x,
            u_x,
            l_y,
            u_y,
            delta,
            Y,
            predicted_Y,
            T,
            inequality,
            candidate_safety_ratio,
            predict_bound,
            modified_h,
            sides,
        )
    return None, None


##############
# Print Tree #
##############
def inorder(t_node: ExprTree | None) -> None:
    """
    A utility function to log inorder traversal

    :param t_node: ExprTree node
    :return: None
    """
    if t_node is not None:
        inorder(t_node.left)
        logger.debug(f"{t_node.value}")
        inorder(t_node.right)
