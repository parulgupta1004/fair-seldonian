from .bounds import eval_math_bound
from .expression_tree import ExprTree as _BaseExprTree
from .expression_tree import is_func, is_mod, is_operator
from .inequalities import eval_estimate, eval_func_bound


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
    rev_polish_notation = rev_polish_notation.split(" ")
    stack = []
    for element in rev_polish_notation:
        if not is_operator(element) and not is_mod(element):
            t = ExprTree(element)
            stack.append(t)
        else:
            if is_mod(element):
                t = ExprTree(element)
                t1 = None
                t2 = stack.pop()
            else:
                t = ExprTree(element)
                t1 = stack.pop()
                t2 = stack.pop()
            t.right = t1
            t.left = t2
            stack.append(t)
    t = stack.pop()
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
    if t_node is not None:
        x = eval_expr_tree(t_node.left, Y, predicted_Y, T)
        y = eval_expr_tree(t_node.right, Y, predicted_Y, T)
        if x is None:
            if is_func(t_node.value):
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
                return eval_estimate(t_node.value, Y, predicted_Y, T)
            elif is_mod(t_node.value):
                return abs(float(x))
            return None
    return None


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
        if l_x is None and u_x is None:
            if is_func(t_node.value):
                return eval_func_bound(
                    t_node.value,
                    Y,
                    predicted_Y,
                    T,
                    t_node.delta,
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
            if t_node.value == "+":
                return eval_math_bound(l_x, u_x, l_y, u_y, "+")
            elif t_node.value == "-":
                return eval_math_bound(l_x, u_x, l_y, u_y, "-")
            elif t_node.value == "*":
                return eval_math_bound(l_x, u_x, l_y, u_y, "*")
            elif t_node.value == "^":
                return eval_math_bound(l_x, u_x, l_y, u_y, "^")
            elif t_node.value == "/":
                return eval_math_bound(l_x, u_x, l_y, u_y, "/")
            elif is_func(t_node.value):
                return eval_func_bound(
                    t_node.value,
                    Y,
                    predicted_Y,
                    T,
                    t_node.delta,
                    inequality,
                    candidate_safety_ratio,
                    predict_bound,
                    modified_h,
                )
            elif is_mod(t_node.value):
                return eval_math_bound(l_x, u_x, l_y, u_y, "abs")
            return None, None
    return None, None


##############
# Print Tree #
##############
def inorder_ext(t_node):
    if t_node is not None:
        inorder_ext(t_node.left)
        print(t_node.value, t_node.delta)
        inorder_ext(t_node.right)
