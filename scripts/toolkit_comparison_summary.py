"""Compare the encodings analytically, structure for structure.

Part of the toolkit head-to-head; run the three scripts in this order:
``toolkit_comparison_toolkit_side.py`` (needs the reference ``seldonian``
package), then ``toolkit_comparison_our_side.py``, then
``toolkit_comparison_summary.py``. They hand data to each other through
``exp/h2h/`` beside the repo, which is gitignored.
"""

import math
import warnings
from pathlib import Path

import numpy as np

WORK_DIR = Path(__file__).resolve().parents[1] / "exp" / "h2h"
WORK_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = WORK_DIR / "h2h_data.npz"
TOOLKIT_FILE = WORK_DIR / "h2h_toolkit.json"

warnings.filterwarnings("ignore")
d = np.load(DATA_FILE, allow_pickle=True)
T, Y, p = d["T"].astype(str), d["Y"], d["p"]
delta = 0.05


def hoef(n, dlt, r=1.0):
    return r * math.sqrt(math.log(1 / dlt) / (2 * n))


print("Demographic parity, |P(Yhat=1|A) - P(Yhat=1|B)| - 0.1, n=20,000, delta=0.05")
print("All rows Hoeffding, so the comparison is structure-for-structure.\n")
for label, nsplit, nnodes in [
    ("cell encoding, naive tree (8 ways)", delta / 8, 4),
    ("cell encoding, standard tree (8 ways)", delta / 8, 4),
    ("rate encoding, standard tree (4 ways)", delta / 4, 2),
]:
    nA, nB = (T == "A").sum(), (T == "B").sum()
    w = (hoef(nA, nsplit) + hoef(nB, nsplit)) * (2 if nnodes == 4 else 1)
    print(f"  {label:<42} {w:.5f}")
nA, nB = (T == "A").sum(), (T == "B").sum()
aff = math.sqrt(0.5 * math.log(1 / (delta / 2)) * (1.0 / nA + 1.0 / nB))
rate = hoef(nA, delta / 4) + hoef(nB, delta / 4)
cell = 2 * (hoef(nA, delta / 8) + hoef(nB, delta / 8))
print(f"  {'affine forms':<42} {aff:.5f}")
vs_rate, vs_cell = 100 * (1 - aff / rate), 100 * (1 - aff / cell)
print(
    f"\n  affine vs rate encoding (the toolkit's):  {vs_rate:.1f}% tighter, "
    f"{(rate / aff) ** 2:.2f}x data"
)
print(
    f"  affine vs cell encoding (ours):          {vs_cell:.1f}% tighter, "
    f"{(cell / aff) ** 2:.2f}x data"
)
print("\n  toolkit's own default is the t-test, which on this data gives 0.01307 -")
print(
    f"  numerically tighter than affine-Hoeffding ({aff:.5f}) "
    "but NOT distribution-free."
)
