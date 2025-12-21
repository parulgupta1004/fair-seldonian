Getting Started
===============

This page is a user guide for the developers to quickly
be able to use FairSeldonian Python library in their own project 
or, extending/enhancing the codebase to enhance the framework.

Pre-requisites
---------------
Library pre-requisites

* sklearn - machine learning model implementations like logistic regression.

* matplotlib - visualizations of the plots.

* numpy - handling data.

* pandas - handling data in the form of dataframes.

* ray - for parallelisation of the algorithm.

* torch - for tensors used in the codebase.

Quick start
-----------
The complete code resides in root folder.
To run the experiment, you need to amend configurations in `main.py`.


Go to terminal or any IDE and run that file using the following command:

.. code-block::

    python main.py <mode>

The default mode is set as `base`. To understand other modes present in the code, refer to the `Variants` section of this documentation.

Configuration
-------------
The user must setup the following things to make full use of this framework:

- **Configuration of the experiment structure** : In the `main.py`, the user must configure the values for number of threads, number of fractions, number of trials per thread, test:train ratio of the dataset, etc.

- **Configure logistic regression functions** : The user must provide the values of delta (significance level), inequality type (currently supports Hoeffding inequality and t-test), fairness constraint (currently, it supports base variables as real number, True positive, true negative, false positive and false negative; binary operators: +, -, *, / and unary operator: absolute) and candidate-to-safety ratio in this file. In addition to this, the user is expected to replace `simple_logistic` with their model, and provide predict function for their model.


Collaboration
-------------
Please feel free to contribute to this code base. For any queries, contact https://www.linkedin.com/in/parulgupta04/
