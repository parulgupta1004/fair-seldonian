Get started
===========

Install the library, run a first constrained model, and learn to read the
guarantee it gives you.

.. code-block:: bash

   pip install fair-seldonian

Python 3.10 or later. The core install pulls in NumPy, pandas, scikit-learn,
SciPy and PyTorch; ``[plots]`` adds matplotlib for the figures and
``[notebook]`` adds JupyterLab for the example notebooks.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: What it does and why
      :link: intro
      :link-type: doc

      The problem Seldonian algorithms solve, what "high-confidence" means here,
      and why a refusal is a feature rather than a failure.

   .. grid-item-card:: Install and run
      :link: quickstart
      :link-type: doc

      Installation, a worked example end to end, reading ``QSAResult`` and its
      diagnostics, and how to extend the framework to your own model.

.. toctree::
   :maxdepth: 2

   intro
   quickstart
