from .expression_tree import ExprTree as _BaseExprTree
from .expression_tree import (
    _eval_node_bounds,
    construct_expr_tree_base,
    eval_expr_tree_base,
    is_func,
)


class ExprTree(_BaseExprTree):
    """
    Extended expression tree node with delta tracking
    """

    def add_delta(self, delta):
        self.delta = delta


def construct_expr_tree(rev_polish_notation, delta, check_bound, check_constant):
    """
    Returns root of constructed tree for given postfix expression

    :param rev_polish_notation: string with space as delimiter ' '
    :return: ExprTree node
    """
    t = construct_expr_tree_base(rev_polish_notation, node_class=ExprTree)
    configure_delta(t, delta, check_bound, check_constant)
    return t


def configure_delta(t_node, delta, check_bound, check_constant):
    if check_constant:
        add_deltas_constant(t_node, delta)
    else:
        add_deltas(t_node, delta)
    if check_bound:
        hash_map = {}
        check_node_dup(t_node, hash_map)
        change_deltas(t_node, hash_map)


def add_deltas_constant(t_node, delta):
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
            elif is_constant(t_node.left.value):
                child_delta_right = delta
            else:
                child_delta_right = delta / 2
            add_deltas_constant(t_node.right, child_delta_right)


def add_deltas(t_node, delta):
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


def check_node_dup(t_node, hash_map):
    if t_node is not None:
        check_node_dup(t_node.left, hash_map)
        if is_func(t_node.value):
            if t_node.value in hash_map:
                list_of_delta = hash_map[t_node.value]
            else:
                list_of_delta = []
            list_of_delta.append(t_node.delta)
            hash_map[t_node.value] = list_of_delta
        check_node_dup(t_node.right, hash_map)


def is_constant(t_node_value):
    try:
        float(t_node_value)
        return True
    except Exception:
        return False


def change_deltas(t_node, hash_map):
    for k, v in hash_map.items():
        if len(v) > 1:
            change_delta_value(t_node, k, sum(v))


def change_delta_value(t_node, element, delta):
    if t_node is not None:
        change_delta_value(t_node.left, element, delta)
        if t_node.value == element:
            t_node.delta = delta
        change_delta_value(t_node.right, element, delta)


#################
# Evaluate tree #
#################
def eval_expr_tree(t_node, Y=None, predicted_Y=None, T=None):
    return eval_expr_tree_base(t_node, Y, predicted_Y, T)


##########################
# Evaluate conf interval #
##########################
def eval_expr_tree_conf_interval(
    t_node,
    Y,
    predicted_Y,
    T,
    inequality,
    candidate_safety_ratio,
    predict_bound,
    modified_h,
):
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
        )
    return None, None


##############
# Print Tree #
##############
def inorder_ext(t_node):
    if t_node is not None:
        inorder_ext(t_node.left)
        print(t_node.value, t_node.delta)
        inorder_ext(t_node.right)
