from .bounds import eval_math_bound
from .inequalities import eval_estimate, eval_func_bound


####################
# Construct Parser #
####################
class expr_tree:
    """
    An expression tree node of the constraint tree
    """

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def isOperator(element):
    if (
        element == "+"
        or element == "-"
        or element == "*"
        or element == "/"
        or element == "^"
    ):
        return True
    return False


def isMod(element):
    if element == "abs":
        return True
    return False


def isFunc(element):
    if (
        element.startswith("FP")
        or element.startswith("FN")
        or element.startswith("TP")
        or element.startswith("TN")
    ):
        return True
    return False


def construct_expr_tree_base(rev_polish_notation):
    """
    Returns root of constructed tree for given postfix expression

    :param rev_polish_notation: string with space as delimiter ' '
    :return: expr_tree node
    """
    rev_polish_notation = rev_polish_notation.split(" ")
    stack = []
    for element in rev_polish_notation:
        if not isOperator(element) and not isMod(element):
            t = expr_tree(element)
            stack.append(t)
        else:
            if isMod(element):
                t = expr_tree(element)
                t1 = None
                t2 = stack.pop()
            else:
                t = expr_tree(element)
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
def eval_expr_tree_base(t_node, Y, predicted_Y, T):
    """
    A utility function to evaluate estimate of the expression tree

    :param t_node: expr_tree node
    :param Y: pandas::Series
    :param predicted_Y: tensor
    :param T: pandas::Series
    :return: estimate value: float
    """
    if t_node is not None:
        x = eval_expr_tree_base(t_node.left, Y, predicted_Y, T)
        y = eval_expr_tree_base(t_node.right, Y, predicted_Y, T)
        if x is None:
            if isFunc(t_node.value):
                return eval_estimate(t_node.value, Y, predicted_Y, T)
            return float(t_node.value)
        elif y is None:
            if isMod(t_node.value):
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
            elif isFunc(t_node.value):
                return eval_estimate(t_node.value, Y, predicted_Y, T)
            elif isMod(t_node.value):
                return abs(float(x))
            return None
    return None


##########################
# Evaluate conf interval #
##########################
def eval_expr_tree_conf_interval_base(
    t_node,
    Y,
    predicted_Y,
    T,
    delta,
    inequality,
    candidate_safety_ratio,
    predict_bound,
    modified_h,
):
    """
    To evaluate confidence interval of the expression tree

    :param t_node: expr_tree node
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
        if l_x is None and u_x is None:
            if isFunc(t_node.value):
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
            if isMod(t_node.value):
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
            elif isFunc(t_node.value):
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
            elif isMod(t_node.value):
                return eval_math_bound(l_x, u_x, l_y, u_y, "abs")
            return None, None
    return None, None


##############
# Print Tree #
##############
def inorder(t_node):
    """
    A utility function to print inorder traversal

    :param t_node: expr_tree node
    :return: None
    """
    if t_node is not None:
        inorder(t_node.left)
        print(t_node.value, end=" ")
        inorder(t_node.right)
