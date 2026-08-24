Algorithm Variants
==================

The framework implements several optimizations to the base Seldonian algorithm
[Thomas2019]_ that tighten confidence bounds, leading to improved solution rates
and objective performance. Each variant can be selected via the
``seldonian_type`` argument.

This axis is independent of the :doc:`confidence inequality <inequalities>`: the
inequality sets how wide a single interval is, the variant sets how the intervals
are combined. Every pairing is accepted.

.. fs-variant-table::

.. _variant-base:

Baseline QSA (``base``)
-----------------------

The baseline algorithm uses uniform delta splitting and the standard Hoeffding
bound [Hoeffding1963]_ for predicting the safety test outcome during candidate
selection.

**Predicted confidence interval.** During candidate selection, the upper bound
on :math:`g(\theta)` is estimated as:

.. math::

   \hat{p} \pm 2\sqrt{\frac{\ln(1/\delta)}{2\,|\mathcal{D}_s|}}

where :math:`|\mathcal{D}_s| = (1 - r) \cdot |\mathcal{D}|` and :math:`r` is the
candidate ratio.

**Delta splitting.** At each binary operator in the constraint expression tree,
:math:`\delta` is split uniformly:

.. math::

   \delta_{\text{left}} = \delta_{\text{right}} = \frac{\delta}{2}

.. code-block:: bash

   uv run python scripts/run_paper_experiments.py --out exp/paper --only base

.. _variant-mod:

Modified Confidence Interval (``mod``)
--------------------------------------

The baseline doubles the Hoeffding term to account for estimation error in
*both* the candidate estimate and the safety bound. This is conservative: the
two sources of error have different sample sizes.

The modified bound decomposes the interval into separate terms:

.. math::

   \hat{p} \pm \underbrace{\sqrt{\frac{\ln(1/\delta)}{2\,|\mathcal{D}_c|}}}_{\text{candidate error}}
   + \underbrace{\sqrt{\frac{\ln(1/\delta)}{2\,|\mathcal{D}_s|}}}_{\text{safety error}}

**When does this help?** When :math:`|\mathcal{D}_c| \neq |\mathcal{D}_s|`, the
decomposed form yields a tighter interval than doubling the safety-only term.
The improvement is most pronounced at extreme candidate ratios.

.. code-block:: bash

   uv run python scripts/run_paper_experiments.py --out exp/paper --only mod

.. _variant-const:

Constant-Aware Delta Allocation (``const``)
-------------------------------------------

In the baseline, delta is split equally at every binary operator node. However,
when one child is a numeric constant (e.g., ``0.25``), its value is exact — no
confidence interval is needed. The full :math:`\delta` can therefore be allocated
to the non-constant child.

.. figure:: images/const.png
   :align: center
   :width: 80%

   Comparison of uniform vs. constant-aware delta allocation. When a child node
   is a constant, the full :math:`\delta` passes through to the variable subtree.

**Rule.** At a binary operator node with :math:`\delta`:

.. math::

   \delta_{\text{child}} =
   \begin{cases}
   \delta & \text{if the sibling is a constant} \\
   \delta / 2 & \text{otherwise}
   \end{cases}

.. code-block:: bash

   uv run python scripts/run_paper_experiments.py --out exp/paper --only const

.. _variant-bound:

Union Bound Optimization (``bound``)
-------------------------------------

A fairness constraint may reference the same base variable (e.g., ``TP(1)``)
multiple times. The baseline treats each occurrence independently, assigning
each its own :math:`\delta_i`. By Boole's inequality (the union bound)
[Bonferroni1936]_, a single confidence interval with
:math:`\delta_{\text{sum}} = \sum_i \delta_i` covers *all* occurrences
simultaneously.

**Example.** Suppose ``TP(1)`` appears three times in the constraint tree with
allocated deltas :math:`\delta/2`, :math:`\delta/4`, and :math:`\delta/8`. Instead
of computing three separate intervals, a single interval is computed with:

.. math::

   \delta_{\text{sum}} = \frac{\delta}{2} + \frac{\delta}{4} + \frac{\delta}{8}
   = \frac{7\delta}{8}

This yields a wider effective :math:`\delta` and hence a *tighter* confidence
interval for each occurrence, since
:math:`\sqrt{\ln(1/\delta_{\text{sum}})} < \sqrt{\ln(1/\delta_i)}` for each :math:`\delta_i < \delta_{\text{sum}}`.

.. figure:: images/bound-no.png
   :align: center
   :width: 80%

   Without union bound optimization: each occurrence of the same variable
   uses a separate, smaller :math:`\delta`.

.. figure:: images/bound-yes.png
   :align: center
   :width: 80%

   With union bound optimization: all occurrences share a combined
   :math:`\delta`, yielding tighter bounds.

.. code-block:: bash

   uv run python scripts/run_paper_experiments.py --out exp/paper --only bound

.. _variant-opt:

All Optimizations (``opt``)
----------------------------

Combines the modified confidence interval (:ref:`variant-mod`), constant-aware
delta allocation (:ref:`variant-const`), and union bound optimization
(:ref:`variant-bound`) for the tightest bounds.

.. code-block:: bash

   uv run python scripts/run_paper_experiments.py --out exp/paper --only opt

.. _variant-affine:

Affine-Form Compilation (``affine``)
------------------------------------

The variants above all keep the same underlying strategy: wrap a confidence
interval around every node of the constraint tree and combine them with interval
arithmetic. That is sound but loose for two compounding reasons. Interval
arithmetic assumes the worst about how sub-expressions relate, even when they are
means over *disjoint* groups and therefore independent; and every leaf occurrence
spends its own slice of :math:`\delta`.

For constraints built only from ``+``, ``-``, scaling by a constant and ``abs``
— which covers the standard gap-based fairness definitions — neither cost is
necessary. Such an expression can be rewritten *exactly* as a maximum of finitely
many affine forms in the base variables:

.. math::

   g(\theta) = \max_k \left( c_k + \sum_v a_{k,v} z_v \right)

**Why one interval per form suffices.** All cells of a single group are means
over the same rows, so for group :math:`g` the partial sum
:math:`\sum_v a_v z_v` is itself the mean over that group's samples of the scalar
:math:`w_i = \sum_v a_v x_i^{(v)}`. Distinct groups are disjoint, so conditional
on the group counts their means are independent and Hoeffding applies to the
weighted sum directly:

.. math::

   \text{half-width} = \sqrt{\tfrac{1}{2}\ln(1/\delta')\sum_g r_g^2 / n_g}

with :math:`r_g` the a-priori range of :math:`w_i` within group :math:`g`. Only
:math:`K` slices of :math:`\delta` are spent — one per form — instead of one per
leaf occurrence.

**Worked example.** The constraint ``|TP(0) - TP(1)| - 0.2 TP(1) <= 0`` compiles
to ``max(TP(0) - 1.2 TP(1), -TP(0) + 0.8 TP(1))``: two forms rather than three
leaf intervals. Measured slack over the true value falls by 50.0% across sample
sizes from 10k to 160k and several parameter vectors. The ratio barely moves,
because the saving comes from the delta split and the independence of the groups
rather than from the data. Since a Hoeffding half-width scales as
:math:`1/\sqrt{n}`, halving it is worth roughly 4x the data.

**Limits.** Anything outside that fragment raises
:class:`~fair_seldonian.constraints.affine.NotAffine`. Division is the common
case: :func:`~fair_seldonian.constraints.fairness.equal_opportunity` compares
true-positive *rates*, which divide by a per-group base rate, so it falls back to
interval arithmetic. Writing the constraint over the ``TPR(g)`` primitive instead
of an explicit ratio keeps it inside the fragment.

.. code-block:: bash

   uv run python scripts/run_paper_experiments.py --out exp/paper --only affine

See :mod:`fair_seldonian.constraints.affine` for the compiler itself.

