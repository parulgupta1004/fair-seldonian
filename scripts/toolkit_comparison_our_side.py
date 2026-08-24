import json
import warnings

import numpy as np
import pandas as pd
import torch

import fair_seldonian.models.logistic_regression as LR
from fair_seldonian.config import SeldonianConfig
from fair_seldonian.constraints.expression_tree import (
    construct_expr_tree_base,
    eval_expr_tree_base,
)
from fair_seldonian.constraints.inequalities import Inequality

warnings.filterwarnings("ignore")
d = np.load("/tmp/h2h_data.npz", allow_pickle=True)
T, Y, p = d["T"].astype(str), d["Y"], d["p"]
tk = json.load(open("/tmp/h2h_toolkit.json"))

Ys, Ts, preds = pd.Series(Y), pd.Series(T), torch.tensor(p)
LR.predict = lambda th, t1, X: preds  # feed identical probabilities
C = "TP(A) FP(A) + TP(B) FP(B) + - abs 0.1 -"
truth = float(eval_expr_tree_base(construct_expr_tree_base(C), Ys, preds, Ts))

tt = dict(constraint=C, delta=0.05, candidate_ratio=0.60, inequality=Inequality.T_TEST)
res = {
    "ours, naive tree (t-test)": LR.ghat_base(
        None, None, None, Ys, Ts, False, None, False, SeldonianConfig(**tt)
    ),
    "ours, standard tree (t-test)": LR.ghat_extend(
        None, None, None, Ys, Ts, False, None, True, True, True, SeldonianConfig(**tt)
    ),
    "ours, affine (t-test n/a; Hoeffding)": LR.ghat_affine(
        None,
        None,
        None,
        Ys,
        Ts,
        None,
        SeldonianConfig(constraint=C, delta=0.05, candidate_ratio=0.60),
    ),
}
tk_upper = float(tk["upper"])
print(f"\ntrue g = {truth:.6f}\n")
print(f"{'bound':<40} {'U(g)':>10} {'slack':>10}")
toolkit_label = "Seldonian Toolkit (t-test)"
print(f"{toolkit_label:<40} {tk_upper:>10.6f} {tk_upper - truth:>10.6f}")
for k, v in res.items():
    v = float(v)
    print(f"{k:<40} {v:>10.6f} {v - truth:>10.6f}")
std = float(res["ours, standard tree (t-test)"])
aff = float(res["ours, affine (t-test n/a; Hoeffding)"])
apart = abs(std - tk_upper) / tk_upper * 100
tighter = 100 * (1 - (aff - truth) / (tk_upper - truth))
print(f"\n  our 'standard' vs toolkit: {apart:.2f}% apart")
print(f"  affine vs toolkit slack  : {tighter:.1f}% tighter")
