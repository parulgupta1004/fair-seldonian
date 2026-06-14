from __future__ import annotations

import logging
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


def is_func(element: str) -> bool:
    return element.startswith(("FP", "FN", "TP", "TN"))


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
) -> tuple[Bound | None, Bound | None]:
    if l_x is None and u_x is None:
        if is_func(t_node.value):
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
            )
        return float(t_node.value), float(t_node.value)
    elif l_y is None and u_y is None:
        if is_mod(t_node.value):
            return eval_math_bound(l_x, u_x, l_y, u_y, "abs")
        return None, None
    else:
        if is_operator(t_node.value):
            return eval_math_bound(l_x, u_x, l_y, u_y, t_node.value)
        elif is_func(t_node.value):
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
            )
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
    :return: upper and lower bound of the estimate of the constraint
    """
    if t_node is not None:
        if t_node.right is not None and t_node.right.value is not None:
            child_delta = delta / 2
        else:
            child_delta = delta
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
