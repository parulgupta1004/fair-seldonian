Getting Started
===============

Installation
------------

**Requirements:** Python 3.10 or later.

The recommended installation uses `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/parulgupta1004/fair-seldonian.git
   cd fair-seldonian
   uv sync

To include optional dependencies for experiments (Ray) and visualization (matplotlib):

.. code-block:: bash

   uv sync --extra experiments --extra plots

Alternatively, with pip:

.. code-block:: bash

   pip install -e .
   pip install -e ".[experiments,plots]"

Dependencies
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Package
     - Purpose
     - Required
   * - NumPy
     - Array operations and linear algebra
     - Yes
   * - pandas
     - Tabular data handling
     - Yes
   * - PyTorch
     - Tensor operations and automatic differentiation
     - Yes
   * - scikit-learn
     - Baseline logistic regression model
     - Yes
   * - SciPy
     - Statistical functions and numerical optimization
     - Yes
   * - matplotlib
     - Visualization of experiment results
     - Optional
   * - Ray
     - Distributed parallel experiment execution
     - Optional

Running Experiments
-------------------

Experiments are executed via the command line. The ``seldonian_type`` argument
selects the algorithm variant (see :doc:`variants` for details):

.. code-block:: bash

   uv run python -m fair_seldonian.experiments.runner <mode>

where ``<mode>`` is one of ``base``, ``mod``, ``bound``, ``const``, or ``opt``.

Results are saved as ``.npz`` files in ``exp/exp_<mode>/bin/``. To aggregate
results and generate plots:

.. code-block:: bash

   uv run python -m fair_seldonian.experiments.plots

The generated plots show three metrics as a function of training set size:

1. **Log loss** — primary objective performance.
2. **Probability of solution** — fraction of trials where a solution was found.
3. **Probability of constraint violation** — fraction of trials where
   :math:`g(\theta) > 0` on test data (should remain below :math:`\delta`).

Library Usage
-------------

The framework can also be used programmatically:

.. code-block:: python

   from fair_seldonian.algorithms import QSA
   from fair_seldonian.models import simple_logistic, eval_ghat
   from fair_seldonian.data import get_data, data_split

   # Generate synthetic data with configurable group ratios
   data = get_data(N=10000, features=5, t_ratio=0.4,
                   tp0_ratio=0.4, tp1_ratio=0.6, random_seed=42)

   # Split into train and test sets (80/20)
   X_test, Y_test, T_test, X_train, Y_train, T_train = data_split(
       frac=0.5, All=data, random_state=1, mTest=0.2)

   # Run the Quasi-Seldonian Algorithm with all optimizations
   theta, theta1, passed = QSA(
       X_train, Y_train, T_train,
       seldonian_type="opt",
       init_sol=None, init_sol1=None,
   )

   if passed:
       # Evaluate the constraint on held-out test data
       violation = eval_ghat(theta, theta1,
                             X_test, Y_test, T_test, "opt")
       print(f"Constraint upper bound on test data: {violation:.6f}")
   else:
       print("No Solution Found — constraint could not be satisfied.")

Configuration
-------------

The framework is configured through module-level variables in
:mod:`fair_seldonian.models.logistic_regression`:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``delta``
     - 0.05
     - Significance level :math:`\delta`; the constraint holds with
       probability :math:`\geq 1 - \delta`
   * - ``ineq``
     - Hoeffding
     - Concentration inequality used for bound computation
       (:class:`~fair_seldonian.constraints.inequalities.Inequality`)
   * - ``rev_polish_notation``
     - See below
     - Fairness constraint in reverse Polish notation
   * - ``candidate_ratio``
     - 0.40
     - Fraction of training data allocated to the candidate set

The default constraint string ``TP(1) TP(0) - abs 0.25 TP(1) * -`` encodes
a relaxed equalized opportunity condition (see :doc:`intro` for details).

Extending the Framework
-----------------------

To use a custom model, replace the following functions in
:mod:`fair_seldonian.models.logistic_regression`:

- :func:`~fair_seldonian.models.logistic_regression.predict` — returns
  :math:`P(Y=1 \mid X, \theta)` as a tensor.
- :func:`~fair_seldonian.models.logistic_regression.simple_logistic` — trains the
  base model and returns initial parameter values.
- :func:`~fair_seldonian.models.logistic_regression.fHat` — computes the primary
  objective function.

The constraint specification (``rev_polish_notation``) can be modified to encode
any fairness condition expressible in terms of ``TP``, ``FP``, ``TN``, ``FN``
rates across groups.
