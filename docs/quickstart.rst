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

   uv run python scripts/run_paper_experiments.py --out exp/paper --only <mode>

where ``<mode>`` is one of ``base``, ``mod``, ``bound``, ``const``, ``opt`` or
``affine``. Omit ``--only`` to run every variant.

Each variant writes ``summary.csv`` and its four result panels to
``exp/paper/<mode>/``; the run, the aggregation and the plots all happen in that
one command. Use ``--trials`` to set the repetitions per dataset size (default
40) and ``--jobs`` to run trials in parallel.

The generated plots show three metrics as a function of training set size:

1. **Log loss** — primary objective performance.
2. **Probability of solution** — fraction of trials where a solution was found.
3. **Probability of constraint violation** — fraction of trials where
   :math:`g(\theta) > 0` on test data (should remain below :math:`\delta`).

Library Usage
-------------

The framework can also be used programmatically:

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :end-before: if __name__

Configuration
-------------

The algorithm is configured via the :class:`~fair_seldonian.config.SeldonianConfig`
dataclass. All parameters have sensible defaults, so no configuration is required
for basic usage.

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
   * - ``inequality``
     - Hoeffding
     - Concentration inequality used for bound computation
       (:class:`~fair_seldonian.constraints.inequalities.Inequality`)
   * - ``constraint``
     - See below
     - Fairness constraint in reverse Polish notation
   * - ``candidate_ratio``
     - 0.40
     - Fraction of training data allocated to the candidate set

The default constraint string ``TP(1) TP(0) - abs 0.25 TP(1) * -`` encodes
a relaxed equalized opportunity condition (see :doc:`intro` for details).

**Example: custom configuration**

.. literalinclude:: ../examples/custom_constraint.py
   :language: python
   :start-at: strict = SeldonianConfig(
   :end-before: evaluate("Stricter

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

The constraint expression (``constraint`` field on
:class:`~fair_seldonian.config.SeldonianConfig`) can be set to any fairness
condition expressible in terms of ``TP``, ``FP``, ``TN``, ``FN`` rates across
groups.
