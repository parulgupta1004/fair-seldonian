Introduction
============

Motivation
----------

As machine learning systems are increasingly deployed in high-stakes domains — hiring,
lending, criminal justice, healthcare — ensuring that these systems do not exhibit
discriminatory behavior is a critical requirement. Standard ML training procedures
optimize predictive accuracy without regard for fairness, and post-hoc auditing
provides no guarantees that a deployed model will satisfy fairness constraints on
future data.

**Seldonian algorithms** [Thomas2019]_ address this gap by providing
*high-confidence guarantees* that a learned model will satisfy user-specified
behavioral constraints. The key insight is that the responsibility for enforcing
fairness is shifted from the user (who audits after deployment) to the algorithm
designer (who builds safety into the training procedure itself).

Problem Formulation
-------------------

Let :math:`\mathcal{D}` denote a dataset of :math:`n` i.i.d. samples, and let
:math:`a(\mathcal{D})` denote the model returned by a learning algorithm :math:`a`.
We define:

- **Primary objective** :math:`f(\theta)`: the quantity to be maximized (e.g.,
  negative log loss).
- **Behavioral constraint** :math:`g(\theta) \leq 0`: a fairness condition that
  must hold with high probability (e.g., bounded disparity in group-level
  true positive rates).
- **Confidence level** :math:`1 - \delta`: the minimum probability with which
  the constraint must be satisfied.

A Seldonian algorithm solves:

.. math::

   \max_\theta \; f(\theta) \quad \text{subject to} \quad
   \Pr\!\left[\, g\!\left(a(\mathcal{D})\right) \leq 0 \,\right] \geq 1 - \delta

If no solution satisfying the constraint can be found with sufficient confidence,
the algorithm returns **No Solution Found** (NSF) rather than a potentially
unsafe model.

Framework Overview
------------------

The Fair-Seldonian framework implements the **Quasi-Seldonian Algorithm (QSA)**, which
proceeds in three stages:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Stage
     - Description
   * - **1. Data Splitting**
     - The training data :math:`\mathcal{D}` is partitioned into a *candidate set*
       :math:`\mathcal{D}_c` and a *safety set* :math:`\mathcal{D}_s` according to
       a configurable ratio (default 40/60).
   * - **2. Candidate Selection**
     - An optimization procedure finds :math:`\theta^*` that maximizes :math:`f(\theta)`
       subject to a *predicted* upper bound on :math:`g(\theta)` computed from
       :math:`\mathcal{D}_c`. This predicted bound accounts for the statistical
       uncertainty that will remain when the safety test uses :math:`\mathcal{D}_s`.
   * - **3. Safety Test**
     - The candidate :math:`\theta^*` is evaluated on :math:`\mathcal{D}_s` using
       a concentration inequality (Hoeffding or Student's t). If the upper bound
       on :math:`g(\theta^*)` is non-positive, the solution is accepted; otherwise,
       **No Solution Found** is returned.

Constraint Specification
------------------------

Behavioral constraints are specified as strings in
`reverse Polish notation <https://en.wikipedia.org/wiki/Reverse_Polish_notation>`_
(RPN) and parsed into expression trees for evaluation. The supported primitives are:

.. fs-basevar-table::

Supported operators: ``+``, ``-``, ``*``, ``/``, ``^``, ``abs``.

The distinction between the three kinds is easy to miss and it matters. A *cell*
such as ``TP(g)`` is a fraction of the whole of group :math:`g` — every row of
the group contributes, and the four cells sum to 1. A *rate* such as ``TPR(g)``
is a mean over only that group's positive rows, so it carries a smaller sample
size and a wider interval. :doc:`fairness_constraints` covers how to choose.

**Example.** The default constraint string::

    TP(1) TP(0) - abs 0.25 TP(1) * -

encodes the infix expression:

.. math::

   \left| \text{TP}(1) - \text{TP}(0) \right| - 0.25 \cdot \text{TP}(1) \leq 0

This requires the gap between the groups' true-positive *cells* to be at most
25% of group 1's cell.

.. note::

   This is **not** equal opportunity, despite the resemblance. Equal opportunity
   compares true-positive *rates* — ``TPR(g)``, conditioned on :math:`Y = 1` —
   and is available as
   :func:`~fair_seldonian.constraints.fairness.equal_opportunity`. The default
   constraint compares cells, which is a different quantity computed over a
   different set of rows.

Confidence Bound Propagation
----------------------------

Evaluating :math:`g(\theta)` requires computing confidence intervals for each
leaf node (e.g., ``TP(1)``) and propagating them through the expression tree
using interval arithmetic. Four concentration inequalities are available, three
of them distribution-free:

.. fs-inequality-table::

**Hoeffding's inequality** [Hoeffding1963]_ is the default. For a bounded random
variable with :math:`n` samples:

.. math::

   \Pr\!\left[\left|\hat{p} - p\right| \geq \epsilon\right]
   \leq 2\exp\!\left(-2n\epsilon^2\right)

Empirical Bernstein and betting both adapt to the variance they observe, and are
tighter than Hoeffding whenever a group's rate is far from :math:`1/2`.
**Student's t** assumes the sample mean is approximately normal — an appeal to
the central limit theorem rather than a finite-sample bound, and what puts the
*quasi* in quasi-Seldonian. :doc:`inequalities` derives each and shows where the
ordering between them changes.

At each internal node of the expression tree, the confidence intervals of the
children are combined according to the rules of interval arithmetic
[Moore1966]_ (see :mod:`fair_seldonian.constraints.bounds`). The framework
provides several optimizations to tighten these bounds, described in
:doc:`variants`.

