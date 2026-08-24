:html_theme.sidebar_secondary.remove: true

Fair-Seldonian
==============

.. rst-class:: fs-tagline

**Train a model that provably satisfies a fairness constraint — or refuses.**

Given a behavioural constraint and a confidence level :math:`\delta`, the
Quasi-Seldonian Algorithm returns a model satisfying the constraint with
probability :math:`\geq 1 - \delta`, or returns **No Solution Found**. It never
returns a model it cannot certify.

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: sd-mb-4

   .. grid-item-card:: Never an unsafe model
      :link: theory
      :link-type: doc

      The safety test runs on data held out from candidate selection, so the
      guarantee is a genuine high-confidence bound rather than a training-set
      measurement.

   .. grid-item-card:: Five fairness definitions
      :link: fairness_constraints
      :link-type: doc

      Demographic parity, equal opportunity, equalized odds, error rate and
      error-rate parity — ready to use, or write your own in the constraint DSL.

   .. grid-item-card:: Tighter bounds
      :link: variants
      :link-type: doc

      Affine-form compilation roughly halves the slack over interval arithmetic,
      worth about 4x the data at a fixed confidence level.

   .. grid-item-card:: Choose your inequality
      :link: inequalities
      :link-type: doc

      Hoeffding, empirical Bernstein and betting intervals are distribution-free;
      Student's t is available for comparison against the literature.

Get started
-----------

.. code-block:: bash

   pip install fair-seldonian

.. code-block:: python

   from fair_seldonian import QSA, SeldonianConfig, data_split, demographic_parity, get_data

   data = get_data(N=20000, features=5, t_ratio=0.5,
                   tp0_ratio=0.4, tp1_ratio=0.6, random_seed=42)
   X_te, Y_te, T_te, X_tr, Y_tr, T_tr = data_split(
       frac=0.6, all_data=data, random_state=1, m_test=0.2)

   config = SeldonianConfig(constraint=demographic_parity(epsilon=0.2), delta=0.05)
   result = QSA(X_tr, Y_tr, T_tr, "opt", None, None, config)

   if result.passed_safety:
       print(f"certified: bound <= {result.diagnostics.safety_upper_bound:+.4f}")
   else:
       print(f"no solution ({result.diagnostics.failure_mode})")

.. seealso::

   :doc:`quickstart` walks through the same example in detail, and
   :doc:`examples/index` has runnable notebooks including one on UCI Adult.

.. list-table::
   :widths: 20 80

   * - **Repository**
     - `github.com/parulgupta1004/fair-seldonian <https://github.com/parulgupta1004/fair-seldonian>`_
   * - **Paper**
     - Thomas et al., *Science* 366 (2019) —
       `doi:10.1126/science.aag3311 <https://www.science.org/doi/10.1126/science.aag3311>`_
   * - **Python**
     - 3.10+

.. note::

   For the foundational work on Seldonian algorithms, see:

   Thomas, P.S., da Silva, B.C., Barto, A.G., Giguere, S., Brun, Y., &
   Brunskill, E. (2019). "Preventing undesirable behavior of intelligent
   machines." *Science*, 366(6468), 999–1004.
   `[DOI] <https://www.science.org/doi/10.1126/science.aag3311>`_
   `[Project site] <https://aisafety.cs.umass.edu>`_

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started
   concepts
   examples/index
   API reference <api/fair_seldonian>
