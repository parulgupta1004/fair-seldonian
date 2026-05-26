Fair-Seldonian
==============

**Fairness-constrained machine learning with high-confidence guarantees.**

Fair-Seldonian is a Python framework implementing the Quasi-Seldonian
Algorithm (QSA) for training ML models that provably satisfy fairness
constraints. Given a behavioral constraint and a confidence level
:math:`\delta`, the algorithm either returns a model satisfying the constraint
with probability :math:`\geq 1 - \delta`, or returns **No Solution Found** —
never an unsafe model.

.. list-table::
   :widths: 20 80

   * - **Documentation**
     - `parulgupta1004.github.io/fair-seldonian <https://parulgupta1004.github.io/fair-seldonian/>`_
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
   :maxdepth: 2
   :caption: User Guide

   intro
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Algorithm Details

   theory
   variants

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/fair_seldonian

.. toctree::
   :caption: Bibliography

   references

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
