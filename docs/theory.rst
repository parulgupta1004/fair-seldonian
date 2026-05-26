Mathematical Background
=======================

This section provides the formal mathematical details underlying the
Fair-Seldonian framework. The core algorithm follows the Quasi-Seldonian
approach introduced in [Thomas2019]_.

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Symbol
     - Definition
   * - :math:`\mathcal{D}`
     - Training dataset of :math:`n` i.i.d. samples :math:`\{(x_i, y_i, t_i)\}_{i=1}^n`
   * - :math:`\mathcal{D}_c, \mathcal{D}_s`
     - Candidate and safety data splits
   * - :math:`\theta`
     - Model parameters
   * - :math:`f(\theta)`
     - Primary objective function (to maximize)
   * - :math:`g(\theta)`
     - Behavioral constraint function (:math:`g(\theta) \leq 0` required)
   * - :math:`\delta`
     - Significance level; constraint holds with probability :math:`\geq 1 - \delta`
   * - :math:`\hat{g}(\theta)`
     - Upper confidence bound on :math:`g(\theta)`
   * - :math:`T`
     - Sensitive attribute (group membership)

Quasi-Seldonian Algorithm
--------------------------

The QSA consists of two computational phases after data splitting.

**Candidate selection.** Find :math:`\theta^*` by solving:

.. math::

   \theta^* = \arg\max_\theta \; f(\theta)
   \quad \text{s.t.} \quad \hat{g}_c(\theta, \mathcal{D}_c) \leq 0

where :math:`\hat{g}_c` is the *predicted* upper bound — an estimate of what
the safety test bound will be, computed using the candidate data. If
:math:`\hat{g}_c(\theta) > 0` for all :math:`\theta` explored by the optimizer,
the objective is penalized:

.. math::

   \tilde{f}(\theta) =
   \begin{cases}
   f(\theta) & \text{if } \hat{g}_c(\theta) \leq 0 \\
   -C - \hat{g}_c(\theta) & \text{otherwise}
   \end{cases}

where :math:`C` is a large constant (default :math:`10{,}000`) that ensures
constraint-violating solutions are strongly disfavored.

**Safety test.** Given :math:`\theta^*`, compute the upper confidence bound
on :math:`g(\theta^*)` using the safety data:

.. math::

   \hat{g}_s(\theta^*, \mathcal{D}_s) \leq 0 \implies \text{accept } \theta^*

.. math::

   \hat{g}_s(\theta^*, \mathcal{D}_s) > 0 \implies \text{No Solution Found}

Delta Splitting
---------------

When the constraint expression tree has binary operators, the confidence level
:math:`\delta` must be split between the left and right subtrees. By Boole's
inequality (the union bound) [Bonferroni1936]_, if each subtree's bound holds
with probability :math:`1 - \delta_i`, the combined bound holds with
probability :math:`1 - \sum_i \delta_i`.

**Uniform splitting** (``base`` mode) assigns :math:`\delta/2` to each child of
every binary operator:

.. math::

   \delta_{\text{left}} = \delta_{\text{right}} = \frac{\delta}{2}

This is conservative: it does not account for constant nodes or repeated
variables. The :doc:`variants` section describes three optimizations that
improve upon this baseline.

Interval Arithmetic
-------------------

Confidence intervals are propagated through the expression tree using standard
interval arithmetic rules [Moore1966]_. For intervals :math:`[l_x, u_x]` and
:math:`[l_y, u_y]`:

**Addition:**

.. math::

   [l_x, u_x] + [l_y, u_y] = [l_x + l_y, \; u_x + u_y]

**Subtraction:**

.. math::

   [l_x, u_x] - [l_y, u_y] = [l_x - u_y, \; u_x - l_y]

**Multiplication:**

.. math::

   [l_x, u_x] \times [l_y, u_y] = \left[\min(S), \; \max(S)\right]

where :math:`S = \{l_x l_y, \; l_x u_y, \; u_x l_y, \; u_x u_y\}`.
The implementation handles all sign combinations (both positive, both negative,
mixed signs) as special cases for efficiency.

**Division:**

.. math::

   [l_x, u_x] \,/\, [l_y, u_y] = [l_x, u_x] \times [1/u_y, \; 1/l_y]
   \quad \text{if } 0 \notin [l_y, u_y]

If :math:`0 \in [l_y, u_y]`, the result is :math:`(-\infty, +\infty)`.

**Absolute value:**

.. math::

   |[l_x, u_x]| =
   \begin{cases}
   [l_x, u_x] & \text{if } l_x \geq 0 \\
   [-u_x, -l_x] & \text{if } u_x \leq 0 \\
   [0, \max(-l_x, u_x)] & \text{if } l_x < 0 < u_x
   \end{cases}

See :mod:`fair_seldonian.constraints.bounds` for the full implementation.

Predicted Bounds
----------------

During candidate selection, the algorithm does not have access to the safety data.
Instead, it *predicts* what the safety test bound will be by accounting for the
statistical uncertainty from both data splits.

**Standard prediction** (``base`` mode) uses a doubled Hoeffding term
[Hoeffding1963]_:

.. math::

   \hat{p} \pm 2\sqrt{\frac{\ln(1/\delta)}{2 \, |\mathcal{D}_s|}}

**Decomposed prediction** (``mod`` mode) separates candidate and safety
estimation error:

.. math::

   \hat{p} \pm \sqrt{\frac{\ln(1/\delta)}{2 \, |\mathcal{D}_c|}}
   + \sqrt{\frac{\ln(1/\delta)}{2 \, |\mathcal{D}_s|}}

The decomposed form yields tighter bounds when :math:`|\mathcal{D}_c|` and
:math:`|\mathcal{D}_s|` differ substantially. See :doc:`variants` for the full
set of optimizations.

