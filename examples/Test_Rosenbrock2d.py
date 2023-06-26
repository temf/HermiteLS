# -*- coding: utf-8 -*-
"""
Solve with HermiteLS
Rosenbrock Function 2-dim

Optimal Solution:
x = (1,1), f(x) = 0

"""

from __future__ import print_function
import numpy as np

import sys
sys.path.append('..')
import pybobyqa_HermiteLS



"""
Please define optimization problem.
"""

# Define objective function
def objfunc(x):
    f = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    return f

# Define gradient function w.r.t. given component
def objfun_derivs(x,component = 0):
    if component == 0:
        deriv = 400*x[0]**3 + 2*x[0] - 400*x[0]*x[1] -2
    elif component == 1:
        deriv = 200*x[1] - 200*x[0]**2
    return deriv


# Define Starting Point
x0 = np.array([1.2, 2.0])

# Define bounds
x_lb = np.array([-10.0, -10.0])
x_ub = np.array([10., 10.])

# Set random seed (for reproducibility)
np.random.seed(0)#1,11

# Define maximum number of function evaluations
max_fcteval = 500

# Define list of known derivatives
known_derivs = [1] # possibilites for n=2:  [],[0],[1],[0,1]

# Number of optimization variables and known derivative directions results in
n = len(x0)
kd = len(known_derivs)

# Define number of interpolation points
# Per default
m_min = max(np.int(n+1),np.int(np.ceil(((n+1)*(n+2))/(2*(1+kd)))))
m_max = np.int(0.5*(n+1)*(n+2))
m_default = max(np.int(2*n+1-kd), m_min)
m = m_default
# or manually
#m=4

# Choose if there is noise or not
noise = False

# Choose Hermite least squares version (currently there is only one version)
version = 1



"""
From above: Function of objective function (incl. derivatives) with or without noise
"""

# Define objective function and derivatives in one
def objfunc_with_derivs(x, known_derivs=[], return_all = False):
    f = objfunc(x)
    derivs_list = np.inf*np.ones((len(x)))
    
    if return_all == False:
        return f
    else:
        if known_derivs == []:
            return f, derivs_list
        else:
            for i in range(len(known_derivs)):
                component = known_derivs[i]
                derivs_list[component] = objfun_derivs(x,component)
            return f, derivs_list


### For tests with noisy data ###
# Define objective function and derivatives in one with noise
def objfunc_with_derivs_noisy(x, known_derivs=[], return_all = False):
    noise = (1.0 + 1e-2 * np.random.normal(size=(1,))[0])
    f = objfunc(x) * noise
    derivs_list = np.inf*np.ones((len(x)))
    
    if return_all == False:
        return f
    else:
        if known_derivs == []:
            return f, derivs_list
        else:
            for i in range(len(known_derivs)):
                component = known_derivs[i]
                derivs_list[component] = objfun_derivs(x,component) * noise
            return f, derivs_list




"""
Solution using Hermite least squares
"""

# No noise
if noise == False:
    soln_local = pybobyqa_HermiteLS.solve(version, objfunc_with_derivs, known_derivs, x0, maxfun=max_fcteval, bounds=(x_lb, x_ub), npt = m, print_progress=False)
    print(soln_local)

# Noise
if noise == True:
    # Noise with regression, but not with PyBOBYQA noisy option
    soln_local = pybobyqa_HermiteLS.solve(version, objfunc_with_derivs_noisy, known_derivs, x0, maxfun=max_fcteval, bounds=(x_lb, x_ub), npt = m, print_progress=False)
    print(soln_local)

    # Noise with regression and with with PyBOBYQA noisy option
    soln_local = pybobyqa_HermiteLS.solve(version, objfunc_with_derivs_noisy, known_derivs, x0, maxfun=max_fcteval, bounds=(x_lb, x_ub), npt = m, print_progress=False, objfun_has_noise=True)
    print(soln_local)


"""
Control Output
"""

print('****** Control output for Hermite least squares test run ******')
print('Optimal solution of the 2d Rosenbrock function is xmin = [1,1], f(xmin) = 0.')
if np.linalg.norm(soln_local.x-np.array([1,1])) < 1e-10 and soln_local.f < 1e-20:
    print('Congratulations, you have found the correct solution.')
else:
    print('Sorry, something went wrong. You did not find the correct solution.')
print('******************************')        









