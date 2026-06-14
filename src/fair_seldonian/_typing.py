"""Internal type aliases shared across :mod:`fair_seldonian`.

These describe the duck-typed values that flow through the constraint and
confidence-bound code. They are imported under ``TYPE_CHECKING`` only, so this
module is never imported at runtime.
"""

from typing import TypeAlias

import numpy as np
import pandas as pd
import torch

# A 1-D dataset column (labels ``Y`` or sensitive attributes ``T``). The library
# accepts either a NumPy array (the experiment pipeline) or a pandas Series.
Array: TypeAlias = np.ndarray | pd.Series

# A numeric confidence-bound value: a Python ``float`` (including ``math.inf``)
# or a scalar ``torch.Tensor`` produced by the estimate functions.
Bound: TypeAlias = float | torch.Tensor
