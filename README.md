# Fair Seldonian

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python framework for enforcing fairness constraints in machine learning using [Seldonian algorithms](https://www.science.org/doi/10.1126/science.aag3311). This project implements the Quasi-Seldonian Algorithm (QSA) with several confidence bound optimization extensions, including Hoeffding and T-test based bounds, constant-aware delta allocation, and duplicate node detection.

Initially developed under the guidance of Dr. Philip S. Thomas, University of Massachusetts Amherst.

## Overview

Seldonian algorithms provide high-confidence guarantees that a learned policy will satisfy user-specified safety constraints. This framework applies that paradigm to fairness: given a behavioral constraint (e.g., bounded disparity in true positive rates across demographic groups), the algorithm either returns a model that satisfies the constraint with high probability or returns "No Solution Found."

The pipeline consists of three stages:

1. **Candidate Selection** -- Optimize the primary objective (e.g., log loss) subject to a predicted upper bound on constraint violation, using the candidate dataset.
2. **Safety Test** -- Evaluate the constraint upper bound on a held-out safety dataset. If the bound is non-positive, the candidate solution is accepted.
3. **Evaluation** -- Measure the deployed model's true constraint violation and objective performance on test data.

## Project Structure

```
src/fair_seldonian/
├── constraints/                # Constraint parsing and confidence bound computation
│   ├── bounds.py               # Interval arithmetic over confidence bounds (+, -, *, /, abs)
│   ├── inequalities.py         # Hoeffding and T-test statistical concentration bounds
│   ├── expression_tree.py      # Base expression tree: parse and evaluate constraint strings
│   └── expression_tree_ext.py  # Extended tree with per-node delta allocation strategies
├── models/                     # Model definitions and constraint evaluation
│   └── logistic_regression.py  # Logistic regression model, objective (fHat), and ghat wrappers
├── algorithms/                 # Core Seldonian algorithms
│   └── qsa.py                  # Quasi-Seldonian Algorithm: candidate selection + safety test
├── data/                       # Data generation and preprocessing
│   └── synthetic.py            # Synthetic dataset generation with configurable group ratios
└── experiments/                # Experiment orchestration and analysis
    ├── runner.py               # Parallel experiment runner (Ray)
    ├── results.py              # Results aggregation from experiment output files
    └── plots.py                # Visualization of objective, solution rate, and failure rate
```

## Installation

Requires Python 3.10+. This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository and install dependencies
git clone https://github.com/parul100495/fair-seldonian.git
cd fair-seldonian
uv sync

# Install with experiment dependencies (includes Ray)
uv sync --extra experiments
```

## Usage

### Running Experiments

The experiment runner supports several Seldonian algorithm variants via the `seldonian_type` argument:

| Mode    | Description                                                    |
|---------|----------------------------------------------------------------|
| `base`  | Standard Hoeffding bound, uniform delta splitting              |
| `mod`   | Modified Hoeffding bound with tighter candidate-safety linking |
| `bound` | Duplicate-aware delta reallocation across shared leaf nodes    |
| `const` | Constant-aware delta allocation (skips splitting for numeric terms) |
| `opt`   | All optimizations combined                                     |

```bash
# Run experiments with the base algorithm
uv run python -m fair_seldonian.experiments.runner base

# Generate plots from experiment results
uv run python -m fair_seldonian.experiments.plots
```

### Using as a Library

```python
from fair_seldonian.algorithms import QSA
from fair_seldonian.models import simple_logistic, eval_ghat
from fair_seldonian.data import get_data, data_split

# Generate synthetic data
data = get_data(N=10000, features=5, t_ratio=0.4, tp0_ratio=0.4, tp1_ratio=0.6, random_seed=42)
X_test, Y_test, T_test, X_train, Y_train, T_train = data_split(frac=0.5, All=data, random_state=1, mTest=0.2)

# Run the Quasi-Seldonian Algorithm
theta, theta1, passed_safety = QSA(X_train, Y_train, T_train, seldonian_type="opt", init_sol=None, init_sol1=None)

if passed_safety:
    print("Solution found, evaluating constraint on test data...")
    print("Upper bound:", eval_ghat(theta, theta1, X_test, Y_test, T_test, "opt"))
```

## Constraint Specification

Constraints are specified as strings in [reverse Polish notation](https://en.wikipedia.org/wiki/Reverse_Polish_notation). The supported primitives are:

- `TP(g)`, `FP(g)`, `TN(g)`, `FN(g)` -- true/false positive/negative rates for group `g`
- Operators: `+`, `-`, `*`, `/`, `^`, `abs`

For example, the default constraint `TP(1) TP(0) - abs 0.25 TP(1) * -` encodes:

```
|TP(1) - TP(0)| - 0.25 * TP(1) <= 0
```

This bounds the absolute difference in true positive rates between groups, scaled by a fraction of the majority group's rate.

## References

> Thomas, P.S., et al. "Preventing undesirable behavior of intelligent machines." *Science* 366.6468 (2019): 999-1004.
> https://www.science.org/doi/10.1126/science.aag3311

## Documentation

Full API documentation is available at: https://parulgupta1004.github.io/fair-seldonian/index.html

## Contributing

Contributions, extensions, and bug reports are welcome. Please open an issue or submit a pull request.

**Author:** [Parul Gupta](https://www.linkedin.com/in/parulgupta04/)
