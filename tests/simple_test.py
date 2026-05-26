from fair_seldonian.constraints.expression_tree import *
from fair_seldonian.constraints.inequalities import Inequality
from fair_seldonian.data.synthetic import *
import pandas as pd
import torch


if __name__ == "__main__":
    rev_polish_notation = "TP(1) TP(0) -"
    delta = 0.1
    r = construct_expr_tree_base(rev_polish_notation)
    # Display infix notation
    print("Infix expression is:")
    inorder(r)
    print("")

    # Evaluate parsed tree
    All = get_data(40, 3, 0.5, 0.5, 0.7, 0)
    T = All.iloc[:, -1]
    X = All.iloc[:, :-2]
    predicted_Y = All.iloc[:, -2]
    Y = pd.Series([1.0 for i in range(T.shape[0])])
    print(T[T.astype(str) == "1"].shape[0] / T.shape[0])

    male_y = Y[T.astype(str) == "1"]
    print(male_y[male_y.astype(str) == "1.0"].shape[0] / male_y.shape[0])
    female_y = Y[T.astype(str) == "0"]
    print(female_y[female_y.astype(str) == "1.0"].shape[0] / female_y.shape[0])

    print(predicted_Y.shape)
    predicted_Y = torch.tensor(predicted_Y)
    print("\nEvaluation: ", eval_expr_tree_base(r, Y, predicted_Y, T))

    # Evaluate conf interval
    l_hoeffding, u_hoeffding = eval_expr_tree_conf_interval_base(r, Y, predicted_Y,
                                                                 T, delta,
                                                                 Inequality.HOEFFDING_INEQUALITY,
                                                                 True, 1, False)
    print("Confidence Interval for expression tree:")
    print(f" Hoeffding Inequality: [{l_hoeffding}, {u_hoeffding}]")
