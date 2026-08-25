Concepts
========

A run of the algorithm is configured along three independent axes: *what* you
constrain, *how wide* each confidence interval is, and *how* those intervals are
combined into a single bound. These pages cover one axis each, on top of the
mathematical background they share.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`book;1.1em;sd-mr-2` Mathematical background
      :link: theory
      :link-type: doc

      Candidate selection and the safety test, delta splitting, interval
      arithmetic, and the predicted bound that steers the optimizer.

   .. grid-item-card:: :octicon:`law;1.1em;sd-mr-2` What you constrain
      :link: fairness_constraints
      :link-type: doc

      The five shipped fairness definitions, how to choose between them, and the
      base variables you write your own constraints over.

   .. grid-item-card:: :octicon:`arrow-both;1.1em;sd-mr-2` How wide each interval is
      :link: inequalities
      :link-type: doc

      Hoeffding, empirical Bernstein, betting and Student's t — what each
      assumes, and which is tighter under what conditions.

   .. grid-item-card:: :octicon:`git-merge;1.1em;sd-mr-2` How intervals combine
      :link: variants
      :link-type: doc

      The ``seldonian_type`` variants, from uniform delta splitting to
      affine-form compilation.

.. seealso::

   The two axes are orthogonal: any ``inequality`` may be paired with any
   ``seldonian_type``.

.. toctree::
   :maxdepth: 2

   theory
   fairness_constraints
   inequalities
   variants
   references
