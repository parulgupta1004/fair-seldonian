import json

import numpy as np
from seldonian.parse_tree.nodes import BaseNode
from seldonian.parse_tree.parse_tree import ParseTree

rng = np.random.default_rng(0)
N = 20_000
T = np.where(rng.random(N) < 0.5, "A", "B")
Y = rng.binomial(1, np.where(T == "A", 0.4, 0.6))
p = np.clip(0.25 + 0.5 * Y + rng.normal(0, 0.18, N), 0.01, 0.99)
np.savez("/tmp/h2h_data.npz", T=T, Y=Y, p=p)

pt = ParseTree(
    delta=0.05,
    regime="supervised_learning",
    sub_regime="classification",
    columns=["A", "B"],
)
pt.build_tree("abs((PR | [A]) - (PR | [B])) - 0.1")
deltas = {k: (d["delta_lower"], d["delta_upper"]) for k, d in pt.base_node_dict.items()}
BaseNode.zhat = lambda self, **kw: p[T == ("A" if "[A]" in self.name else "B")]
for d in pt.base_node_dict.values():
    d["bound_method"] = "ttest"
pt.propagate_bounds(
    branch="safety_test",
    theta=None,
    model=None,
    data_dict={},
    sub_regime="classification",
)
json.dump(
    {
        "upper": float(pt.root.upper),
        "deltas": {k: list(v) for k, v in deltas.items()},
        "n_base_nodes": len(pt.base_node_dict),
    },
    open("/tmp/h2h_toolkit.json", "w"),
)
print(f"toolkit U(g) = {pt.root.upper:.6f}  (deltas {deltas})")
