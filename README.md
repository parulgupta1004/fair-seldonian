# Fair-Seldonian

*Fairness-constrained machine learning with high-confidence guarantees*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Build](https://img.shields.io/github/actions/workflow/status/parulgupta1004/fair-seldonian/ci.yml?branch=master&label=build&logo=github)](https://github.com/parulgupta1004/fair-seldonian/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/parulgupta1004/fair-seldonian/blob/master/LICENSE)
[![Docs](https://img.shields.io/badge/docs-Sphinx-8CA1AF?logo=readthedocs&logoColor=white)](https://parulgupta1004.github.io/fair-seldonian/)
[![Paper](https://img.shields.io/badge/paper-Science%20(2019)-orange)](https://www.science.org/doi/10.1126/science.aag3311)

---

A Python framework implementing the **Quasi-Seldonian Algorithm (QSA)** for training ML models that provably satisfy fairness constraints. Given a behavioral constraint and a confidence level *&delta;*, the algorithm either returns a model satisfying the constraint with probability &ge; 1 &minus; *&delta;*, or returns **No Solution Found** — never an unsafe model.

Built on the Seldonian algorithm framework by [Thomas et al. (2019)](https://www.science.org/doi/10.1126/science.aag3311), with extensions for tighter confidence bounds through constant-aware delta allocation, union bound optimization, and decomposed candidate-safety intervals.

## Quick links

| | |
|---|---|
| **Documentation** | [parulgupta1004.github.io/fair-seldonian](https://parulgupta1004.github.io/fair-seldonian/) |
| **Repository** | [github.com/parulgupta1004/fair-seldonian](https://github.com/parulgupta1004/fair-seldonian) |
| **Paper** | Thomas et al., *Science* 366 (2019) — [doi:10.1126/science.aag3311](https://www.science.org/doi/10.1126/science.aag3311) |

## Installation

```bash
git clone https://github.com/parulgupta1004/fair-seldonian.git
cd fair-seldonian
uv sync                          # core dependencies
uv sync --extra experiments      # + Ray for parallel experiments
uv sync --extra plots            # + matplotlib for visualization
```

Or with pip: `pip install -e ".[experiments,plots]"`

## Usage

```python
from fair_seldonian.algorithms import QSA
from fair_seldonian.models import eval_ghat
from fair_seldonian.data import get_data, data_split

data = get_data(N=10000, features=5, t_ratio=0.4,
                tp0_ratio=0.4, tp1_ratio=0.6, random_seed=42)
X_te, Y_te, T_te, X_tr, Y_tr, T_tr = data_split(
    frac=0.5, All=data, random_state=1, mTest=0.2)

theta, theta1, passed = QSA(X_tr, Y_tr, T_tr, seldonian_type="opt")

if passed:
    print("Upper bound:", eval_ghat(theta, theta1, X_te, Y_te, T_te, "opt"))
else:
    print("No Solution Found")
```

## Algorithm variants

| Mode | Description |
|------|-------------|
| `base` | Standard Hoeffding bound, uniform &delta;-splitting |
| `mod` | Decomposed candidate/safety estimation error |
| `const` | Constant-aware &delta; allocation |
| `bound` | Union bound optimization for repeated variables |
| `opt` | All optimizations combined |

```bash
uv run python -m fair_seldonian.experiments.runner opt
uv run python -m fair_seldonian.experiments.plots
```

## Citation

```bibtex
@software{fair_seldonian,
  author = {Parul Gupta},
  title  = {Fair Seldonian Framework},
  year   = {2020}
}
```

This work builds on:

> Thomas, P.S., da Silva, B.C., Barto, A.G., Giguere, S., Brun, Y., & Brunskill, E. (2019). "Preventing undesirable behavior of intelligent machines." *Science*, 366(6468), 999–1004.

## License

[MIT](LICENSE)

---

**Author:** [Parul Gupta](https://www.linkedin.com/in/parulgupta04/) · Initially developed under the guidance of Dr. Philip S. Thomas, University of Massachusetts Amherst.
