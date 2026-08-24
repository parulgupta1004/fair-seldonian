from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .expression_tree import (
    ROOT_SIDES,
    Sides,
    _eval_node_bounds,
    child_sides,
    construct_expr_tree_base,
    eval_expr_tree_base,
    is_constant,
    is_func,
)

# Left where it is: isort sorts this aliased import by name, placing it between
# ROOT_SIDES and Sides, while ruff keeps aliased imports in a separate trailing
# statement. Both hooks run in pre-commit, so without the skip they rewrite this
# block back and forth forever and no commit can pass.
from .expression_tree import ExprTree as _BaseExprTree  # isort: skip

if TYPE_CHECKING:
    import torch

    from .._typing import Array, Bound
    from .inequalities import Inequality

#: The names this module contributes to the public API. autodoc documents
#: exactly these, so the API reference stays the surface users are meant to
#: call rather than every helper that happens to lack a leading underscore.
__all__ = [
    "ExprTree",
    "change_deltas",
    "construct_expr_tree",
    "eval_expr_tree",
    "eval_expr_tree_conf_interval",
    "inorder_ext",
]

logger = logging.getLogger(__name__)


class ExprTree(_BaseExprTree):
    """
    Extended expression tree node with delta and sidedness tracking
    """

    left: ExprTree | None  # pyrefly: ignore[bad-override-mutable-attribute]
    right: ExprTree | None  # pyrefly: ignore[bad-override-mutable-attribute]
    delta: float
    sides: Sides = ROOT_SIDES

    def add_delta(self, delta: float) -> None:
        self.delta = delta


def construct_expr_tree(
    rev_polish_notation: str, delta: float, check_bound: bool, check_constant: bool
) -> ExprTree:
    """
    Returns root of constructed tree for given postfix expression

    :param rev_polish_notation: string with space as delimiter ' '
    :return: ExprTree node
    """
    t = construct_expr_tree_base(rev_polish_notation, node_class=ExprTree)
    configure_delta(t, delta, check_bound, check_constant)
    return t


def annotate_sides(t_node: ExprTree | None, sides: Sides = ROOT_SIDES) -> None:
    """Record on every node which endpoints of its interval are consumed.

    Sidedness is purely structural, so it is resolved once at construction rather
    than recomputed on every optimizer step. It has to be stored (not derived
    during evaluation) because :func:`change_deltas` needs to reconcile it across
    repeated occurrences of the same base variable - see there.
    """
    if t_node is None:
        return
    t_node.sides = sides
    left_sides, right_sides = child_sides(
        t_node.value,
        sides,
        t_node.left.value if t_node.left is not None else None,
        t_node.right.value if t_node.right is not None else None,
    )
    annotate_sides(t_node.left, left_sides)
    annotate_sides(t_node.right, right_sides)


def configure_delta(
    t_node: ExprTree | None, delta: float, check_bound: bool, check_constant: bool
) -> None:
    if check_constant:
        add_deltas_constant(t_node, delta)
    else:
        add_deltas(t_node, delta)
    annotate_sides(t_node)
    if check_bound:
        hash_map: dict[str, list[tuple[float, Sides]]] = {}
        check_node_dup(t_node, hash_map)
        change_deltas(t_node, hash_map)


def add_deltas_constant(t_node: ExprTree | None, delta: float) -> None:
    if t_node is not None:
        if t_node.left is not None and t_node.left.value is not None:
            if is_constant(t_node.left.value):
                child_delta_left = delta
            elif t_node.right is not None and t_node.right.value is not None:
                if is_constant(t_node.right.value):
                    child_delta_left = delta
                else:
                    child_delta_left = delta / 2
            else:
                child_delta_left = delta
            add_deltas_constant(t_node.left, child_delta_left)
        t_node.add_delta(delta)
        if t_node.right is not None and t_node.right.value is not None:
            if is_constant(t_node.right.value):
                child_delta_right = delta
            elif t_node.left is not None and is_constant(t_node.left.value):
                child_delta_right = delta
            else:
                child_delta_right = delta / 2
            add_deltas_constant(t_node.right, child_delta_right)


def add_deltas(t_node: ExprTree | None, delta: float) -> None:
    if t_node is not None:
        if t_node.left is not None and t_node.left.value is not None:
            if t_node.right is not None and t_node.right.value is not None:
                child_delta_left = delta / 2
            else:
                child_delta_left = delta
            add_deltas(t_node.left, child_delta_left)
        t_node.add_delta(delta)
        if t_node.right is not None and t_node.right.value is not None:
            child_delta_right = delta / 2
            add_deltas(t_node.right, child_delta_right)


def check_node_dup(
    t_node: ExprTree | None, hash_map: dict[str, list[tuple[float, Sides]]]
) -> None:
    if t_node is not None:
        check_node_dup(t_node.left, hash_map)
        if is_func(t_node.value):
            hash_map.setdefault(t_node.value, []).append((t_node.delta, t_node.sides))
        check_node_dup(t_node.right, hash_map)


def change_deltas(
    t_node: ExprTree | None, hash_map: dict[str, list[tuple[float, Sides]]]
) -> None:
    """Collapse repeated occurrences of a base variable onto one shared interval.

    If ``TP(1)`` appears three times with budgets ``d/2``, ``d/4`` and ``d/8``, the
    naive tree treats them as three independent intervals and pays
    ``d/2 + d/4 + d/8`` for them. Building a *single* interval at the summed budget
    ``7d/8`` costs the same failure probability but is narrower than all three, and
    the constraint then sees one consistent value for the variable.

    The sidedness must be merged too, and this is load-bearing: the argument above
    only holds if the occurrences really are one interval. Two occurrences with the
    same delta but different sidedness would produce two *different* widths, hence
    two failure events, and the total would silently become ``2 * 7d/8``. Taking the
    union of the endpoint requirements makes every occurrence identical.
    """
    for element, occurrences in hash_map.items():
        if len(occurrences) > 1:
            merged_delta = sum(delta for delta, _ in occurrences)
            merged_sides = (
                any(sides[0] for _, sides in occurrences),
                any(sides[1] for _, sides in occurrences),
            )
            change_delta_value(t_node, element, merged_delta, merged_sides)


def change_delta_value(
    t_node: ExprTree | None, element: str, delta: float, sides: Sides
) -> None:
    if t_node is not None:
        change_delta_value(t_node.left, element, delta, sides)
        if t_node.value == element:
            t_node.delta = delta
            t_node.sides = sides
        change_delta_value(t_node.right, element, delta, sides)


#################
# Evaluate tree #
#################
def eval_expr_tree(
    t_node: _BaseExprTree | None,
    Y: Array | None = None,
    predicted_Y: torch.Tensor | None = None,
    T: Array | None = None,
) -> Bound | None:
    return eval_expr_tree_base(t_node, Y, predicted_Y, T)


##########################
# Evaluate conf interval #
##########################
def eval_expr_tree_conf_interval(
    t_node: ExprTree | None,
    Y: Array,
    predicted_Y: torch.Tensor,
    T: Array,
    inequality: Inequality,
    candidate_safety_ratio: float | None,
    predict_bound: bool,
    modified_h: bool,
) -> tuple[Bound | None, Bound | None]:
    if t_node is not None:
        l_x, u_x = eval_expr_tree_conf_interval(
            t_node.left,
            Y,
            predicted_Y,
            T,
            inequality,
            candidate_safety_ratio,
            predict_bound,
            modified_h,
        )
        l_y, u_y = eval_expr_tree_conf_interval(
            t_node.right,
            Y,
            predicted_Y,
            T,
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
            t_node.delta,
            Y,
            predicted_Y,
            T,
            inequality,
            candidate_safety_ratio,
            predict_bound,
            modified_h,
            t_node.sides,
        )
    return None, None


##############
# Print Tree #
##############
def inorder_ext(t_node: ExprTree | None) -> None:
    if t_node is not None:
        inorder_ext(t_node.left)
        logger.debug(f"{t_node.value} {t_node.delta}")
        inorder_ext(t_node.right)
