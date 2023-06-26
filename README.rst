Hermite least squares optimization (HermiteLS): a modification of PyBOBYQA for optimization with limited derivative information
====================================================================

HermiteLS is a modification of the PyBOBYQA Version 1.2 framework described below. It allows the usage of some partial derivatives and uses least squares regression insted of interpolation. More details can be found in our paper:

1. Mona Fuhrländer and Sebastian Schöps, `Hermite least squares optimization: a modification of PyBOBYQA for optimization with limited derivative information <https://doi.org/10.1007/s11081-023-09795-y>`_, *Optimization and Engineering*, 2023, `DOI: 10.1007/s11081-023-09795-y <https://doi.org/10.1007/s11081-023-09795-y>`_.

The original Py-BOBYQA is a flexible package for solving bound-constrained general objective minimization, without requiring derivatives of the objective. At its core, it is a Python implementation of the BOBYQA algorithm by Powell, but Py-BOBYQA has extra features improving its performance on some problems (see the papers below for details). Py-BOBYQA is particularly useful when evaluations of the objective function are expensive and/or noisy.

More details about Py-BOBYQA and its enhancements over BOBYQA can be found in our papers:

2. Coralia Cartis, Jan Fiala, Benjamin Marteau and Lindon Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41 [`arXiv preprint: 1804.00154 <https://arxiv.org/abs/1804.00154>`_] 
3. Coralia Cartis, Lindon Roberts and Oliver Sheridan-Methven, `Escaping local minima with derivative-free methods: a numerical investigation <https://doi.org/10.1080/02331934.2021.1883015>`_, *Optimization* (2021). [`arXiv preprint: 1812.11343 <https://arxiv.org/abs/1812.11343>`_] 

Please cite [1,2] when using HermiteLS for local optimization with limited derivative information, and [1,2,3] when using Py-BOBYQA's global optimization heuristic functionality. 

The original paper by Powell is: M. J. D. Powell, The BOBYQA algorithm for bound constrained optimization without derivatives, technical report DAMTP 2009/NA06, University of Cambridge (2009), and the original Fortran implementation is available `here <http://mat.uc.pt/~zhang/software.html>`_.

Documentation
-------------
For the original PyBOBYQA package, see the `online manual <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_.

Citation
--------
If you use HermiteLS in a paper, please cite PyBOBYQA for local (and global) optimization and also:

Fuhrländer, M. and Schöps, S., Hermite least squares optimization: a modification of PyBOBYQA for optimization with limited derivative information, *Optimization and Engineering*, 2023, `DOI: 10.1007/s11081-023-09795-y <https://doi.org/10.1007/s11081-023-09795-y>`_.

PyBOBYQA for local optimization:

Cartis, C., Fiala, J., Marteau, B. and Roberts, L., Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41.

PyBOBYQA for global optimization:

Cartis, C., Roberts, L. and Sheridan-Methven, O., Escaping local minima with derivative-free methods: a numerical investigation, Optimization, (2021). 

Requirements
------------
Py-BOBYQA requires the following software to be installed:

* Python 2.7 or Python 3 (http://www.python.org/)

Additionally, the following python packages should be installed:

* NumPy 1.11 or higher (http://www.numpy.org/)
* SciPy 0.18 or higher (http://www.scipy.org/)
* Pandas 0.17 or higher (http://pandas.pydata.org/), to return the diagnostic information as a DataFrame

Examples
--------
Examples of how to run HermiteLS are given in the examples directory in Github.

License
-------
This algorithm is released under the GNU GPL license.
