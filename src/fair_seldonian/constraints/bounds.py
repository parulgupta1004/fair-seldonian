import math


def is_mod(element):
    if element == "abs":
        return True
    return False


def eval_math_bound(l_x, u_x, l_y=None, u_y=None, operator=None):
    if operator == "+":
        return eval_add_bound(l_x, u_x, l_y, u_y)
    elif operator == "-":
        return eval_subtract_bound(l_x, u_x, l_y, u_y)
    elif operator == "*":
        return eval_multiply_bound(l_x, u_x, l_y, u_y)
    elif operator == "^":
        # power is not supported as of now
        return l_x, u_x
    elif operator == "/":
        return eval_div_bound(l_x, u_x, l_y, u_y)
    elif is_mod(operator):
        return eval_abs_bound(l_x, u_x)
    return None, None


def eval_abs_bound(l_x, u_x):
    """
    :param l_x: lower bound
    :param u_x: upper bound
    :return: lower and upper bound of abs operation
    """
    if l_x is not None and u_x is not None:
        if l_x == math.inf or u_x == math.inf or l_x == -math.inf or u_x == math.inf:
            return 0, math.inf
        elif l_x <= 0 and u_x <= 0:
            return -u_x, -l_x
        elif l_x >= 0 and u_x >= 0:
            return l_x, u_x
        elif l_x <= 0 <= u_x:
            return 0, max(-l_x, u_x)
    return None, None


def eval_div_bound(l_x, u_x, l_y, u_y):
    """
    :param l_x: lower bound of left child
    :param u_x: upper bound of left child
    :param l_y: lower bound of right child
    :param u_y: upper bound of right child
    :return: lower and upper bound of div operation
    """
    if l_x is not None and u_x is not None and l_y is not None and u_y is not None:
        if (l_x == -math.inf and u_x == math.inf) or l_y <= 0 <= u_y:
            # x is unbounded or 0 in y
            return -math.inf, math.inf
        elif l_x >= 0 and l_y >= 0:
            # both x, y are positive
            if u_y == math.inf:
                lower = 0
            else:
                lower = l_x / u_y
            if u_x == math.inf:
                upper = math.inf
            else:
                upper = u_x / l_y
            return lower, upper
        elif u_x <= 0 and u_y <= 0:
            # both x, y are negative
            if l_y == -math.inf:
                lower = 0
            else:
                lower = u_x / l_y

            if l_x == -math.inf:
                upper = math.inf
            else:
                upper = l_x / u_y
            return lower, upper
        elif l_x >= 0 >= u_y:
            # x is positive and y is negative
            if u_x == math.inf:
                lower = -math.inf
            else:
                lower = l_x / u_y
            if l_y == -math.inf:
                upper = 0
            else:
                upper = l_x / l_y
            return lower, upper
        elif u_x <= 0 <= l_y:
            # x is negative and y is positive
            if l_x == -math.inf:
                lower = -math.inf
            else:
                lower = l_x / l_y
            if u_y == math.inf:
                upper = 0
            else:
                upper = u_x / u_y
            return lower, upper
        elif l_x <= 0 <= u_x and l_y >= 0:
            # 0 in x and y is positive
            if l_x == -math.inf:
                lower = -math.inf
            else:
                lower = l_x / l_y
            if u_x == math.inf:
                upper = math.inf
            else:
                upper = u_x / l_y
            return lower, upper
        elif l_x <= 0 <= u_x and u_y <= 0:
            # 0 in x and y is negative
            if u_x == math.inf:
                lower = -math.inf
            else:
                lower = u_x / u_y
            if l_x == -math.inf:
                upper = math.inf
            else:
                upper = l_x / u_y
            return lower, upper
    return None, None


def eval_multiply_bound(l_x, u_x, l_y, u_y):
    """
    :param l_x: lower bound of left child
    :param u_x: upper bound of left child
    :param l_y: lower bound of right child
    :param u_y: upper bound of right child
    :return: lower and upper bound of multiply operation
    """
    if l_x is not None and u_x is not None and l_y is not None and u_y is not None:
        if (l_x == -math.inf and u_x == math.inf) or (
            l_y == -math.inf and u_y == math.inf
        ):
            # one of x, y is unbounded
            return -math.inf, math.inf
        elif l_x >= 0 and l_y >= 0:
            # both x, y are positive
            if u_x == math.inf or u_y == math.inf:
                return l_x * l_y, math.inf
            return l_x * l_y, u_x * u_y
        elif u_x <= 0 and u_y <= 0:
            # both x, y are negative
            if l_x == -math.inf or l_y == -math.inf:
                return u_x * u_y, math.inf
            return u_x * u_y, l_x * l_y
        elif l_x >= 0 >= u_y:
            # x is positive and y is negative
            if u_x == math.inf or l_y == -math.inf:
                return -math.inf, l_x * u_y
            return u_x * l_y, l_x * u_y
        elif u_x <= 0 <= l_y:
            # x is negative and y is positive
            if l_x == -math.inf or u_y == math.inf:
                return -math.inf, u_x * l_y
            return l_x * u_y, u_x * l_y
        elif l_x <= 0 <= u_x and l_y >= 0:
            # 0 in x and y is positive
            if l_x == -math.inf or u_y == math.inf:
                lower = -math.inf
            else:
                lower = l_x * u_y
            if u_x == math.inf or u_y == math.inf:
                upper = math.inf
            else:
                upper = u_x * u_y
            return lower, upper
        elif l_x <= 0 <= u_x and u_y <= 0:
            # 0 in x and y is negative
            if u_x == math.inf or l_y == -math.inf:
                lower = -math.inf
            else:
                lower = u_x * l_y
            if l_x == -math.inf or l_y == -math.inf:
                upper = math.inf
            else:
                upper = l_x * l_y
            return lower, upper
        elif l_x >= 0 and l_y <= 0 <= u_y:
            # 0 in y and x is positive
            if u_x == math.inf or l_y == -math.inf:
                lower = -math.inf
            else:
                lower = u_x * l_y
            if u_x == math.inf or u_y == math.inf:
                upper = math.inf
            else:
                upper = u_x * u_y
            return lower, upper
        elif u_x <= 0 and l_y <= 0 <= u_y:
            # 0 in y and x is negative
            if l_x == -math.inf or u_y == math.inf:
                lower = -math.inf
            else:
                lower = l_x * u_y
            if l_x == -math.inf or l_y == -math.inf:
                upper = math.inf
            else:
                upper = l_x * l_y
            return lower, upper
        elif l_x <= 0 <= u_x and l_y <= 0 <= u_y:
            # 0 in x and 0 in y
            if (
                l_x == -math.inf
                or l_y == -math.inf
                or u_x == math.inf
                or u_y == math.inf
            ):
                # unbounded
                return -math.inf, math.inf
            else:
                return min(l_x * u_y, u_x * l_y), max(u_x * u_y, l_x * l_y)
    return None, None


def eval_subtract_bound(l_x, u_x, l_y, u_y):
    """
    :param l_x: lower bound of left child
    :param u_x: upper bound of left child
    :param l_y: lower bound of right child
    :param u_y: upper bound of right child
    :return: lower and upper bound of subtract operation
    """
    if l_x is not None and u_x is not None and l_y is not None and u_y is not None:
        # lower bound
        if l_x == -math.inf or u_y == math.inf:
            lower = -math.inf
        else:
            lower = l_x - u_y

        # upper bound
        if u_x == math.inf or l_y == -math.inf:
            upper = math.inf
        else:
            upper = u_x - l_y
        return lower, upper
    return None, None


def eval_add_bound(l_x, u_x, l_y, u_y):
    """
    :param l_x: lower bound of left child
    :param u_x: upper bound of left child
    :param l_y: lower bound of right child
    :param u_y: upper bound of right child
    :return: lower and upper bound of add operation
    """
    if l_x is not None and u_x is not None and l_y is not None and u_y is not None:
        # lower bound
        if l_x == -math.inf or l_y == -math.inf:
            lower = -math.inf
        else:
            lower = l_x + l_y

        # upper bound
        if u_x == math.inf or u_y == math.inf:
            upper = math.inf
        else:
            upper = u_x + u_y
        return lower, upper
    return None, None
