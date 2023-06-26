"""
Model
====

Maintain a class which represents an interpolating set, and its corresponding quadratic model.
This class should calculate the various geometric quantities of interest to us.


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

The development of this software was sponsored by NAG Ltd. (http://www.nag.co.uk)
and the EPSRC Centre For Doctoral Training in Industrially Focused Mathematical
Modelling (EP/L015803/1) at the University of Oxford. Please contact NAG for
alternative licensing.

"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from math import sqrt
import numpy as np
import scipy.linalg as LA

from .hessian import to_upper_triangular_vector, to_upper_triangular_vector_MF
from .trust_region import trsbox_geometry
from .util import sumsq, model_value


__all__ = ['Model']


class Model(object):
    def __init__(self, version, known_derivs, npt, x0, f0, derivs0, xl, xu, f0_nsamples, n=None, abs_tol=-1e20, precondition=True, do_logging=True):
        if n is None:
            n = len(x0)
        assert npt >= n + 1, "Require npt >= n+1 for quadratic models"
        assert npt <= (n+1)*(n + 2)//2, "Require npt <= (n+1)(n+2)/2 for quadratic models"
        assert x0.shape == (n,), "x0 has wrong shape (got %s, expect (%g,))" % (str(x0.shape), n)
        assert xl.shape == (n,), "xl has wrong shape (got %s, expect (%g,))" % (str(xl.shape), n)
        assert xu.shape == (n,), "xu has wrong shape (got %s, expect (%g,))" % (str(xu.shape), n)
        self.dim = n
        self.num_pts = npt
        self.do_logging = do_logging
        self.version = version
        self.known_derivs = known_derivs

        # Initialise to blank some useful stuff
        # Interpolation points
        self.xbase = x0.copy()
        self.sl = xl - self.xbase  # lower bound w.r.t. xbase (require xpt >= sl)
        self.su = xu - self.xbase  # upper bound w.r.t. xbase (require xpt <= su)
        self.points = np.zeros((npt, n))  # interpolation points w.r.t. xbase

        # Function values
        self.f_values = np.inf * np.ones((npt, ))  # overall objective value for each xpt
        self.f_values[0] = f0
        self.fderivs = np.zeros((npt,len(x0)))
        self.fderivs[0] = derivs0
        self.kopt = 0  # index of current iterate (should be best value so far)
        self.nsamples = np.zeros((npt,), dtype=np.int)  # number of samples used to evaluate objective at each point
        self.nsamples[0] = f0_nsamples
        self.fbeg = self.f_values[0]  # f(x0), saved to check for sufficient reduction

        # Termination criteria
        self.abs_tol = abs_tol

        # Model information
        self.model_const = 0.0  # constant term for model m(s) = c + J*s
        self.model_grad = np.zeros((n,))  # Jacobian term for model m(s) = c + J*s
        self.model_hess = np.zeros((n,n))

        # Saved point (in absolute coordinates) - always check this value before quitting solver
        self.xsave = None
        self.fsave = None
        self.derivssave = None
        self.gradsave = None
        self.hesssave = None
        self.nsamples_save = None

        # Factorisation of interpolation matrix
        self.precondition = precondition # should the interpolation matrix be preconditioned?
        self.factorisation_current = False
        self.lu = None
        self.piv = None
        self.left_scaling = None
        self.right_scaling = None

    def n(self):
        return self.dim

    def npt(self):
        return self.num_pts

    def xopt(self, abs_coordinates=False):
        return self.xpt(self.kopt, abs_coordinates=abs_coordinates)

    def fopt(self):
        return self.f_values[self.kopt]
    
    def derivsopt(self):
        return self.fderivs[self.kopt]

    def xpt(self, k, abs_coordinates=False):
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        if not abs_coordinates:
            return np.minimum(np.maximum(self.sl, self.points[k, :].copy()), self.su)
        else:
            # Apply bounds and convert back to absolute coordinates
            return self.xbase + np.minimum(np.maximum(self.sl, self.points[k, :]), self.su)

    def fval(self, k):
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        return self.f_values[k]

    def as_absolute_coordinates(self, x):
        # If x were an interpolation point, get the absolute coordinates of x
        return self.xbase + np.minimum(np.maximum(self.sl, x), self.su)

    def xpt_directions(self, include_kopt=True):
        if include_kopt:
            ndirs = self.npt()
        else:
            ndirs = self.npt() - 1
        dirns = np.zeros((ndirs, self.n()))  # vector of directions xpt - xopt, excluding for xopt
        xopt = self.xopt()
        for k in range(self.npt()):
            if not include_kopt and k == self.kopt:
                continue  # skipt
            idx = k if include_kopt or k < self.kopt else k - 1
            dirns[idx, :] = self.xpt(k) - xopt
        return dirns

    def distances_to_xopt(self):
        sq_distances = np.zeros((self.npt(),))
        xopt = self.xopt()
        for k in range(self.npt()):
            sq_distances[k] = sumsq(self.points[k, :] - xopt)
        return sq_distances

    def change_point(self, k, x, f, derivs, allow_kopt_update=True):
        # Update point k to x (w.r.t. xbase), with residual values fvec
        assert 0 <= k < self.npt(), "Invalid index %g" % k

        self.points[k, :] = x.copy()
        self.f_values[k] = f
        self.fderivs[k] = derivs
        self.nsamples[k] = 1
        self.factorisation_current = False

        if allow_kopt_update and self.f_values[k] < self.fopt():
            self.kopt = k
        return

    def swap_points(self, k1, k2):
        self.points[[k1, k2], :] = self.points[[k2, k1], :]
        self.fderivs[[k1, k2], :] = self.fderivs[[k2, k1], :]
        self.f_values[[k1, k2]] = self.f_values[[k2, k1]]
        if self.kopt == k1:
            self.kopt = k2
        elif self.kopt == k2:
            self.kopt = k1
        self.factorisation_current = False
        return

    def add_new_sample(self, k, f_extra):
        # We have resampled at xpt(k) - add this information (f_values is average of all samples)
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        t = np.float(self.nsamples[k]) / np.float(self.nsamples[k] + 1)
        self.f_values[k] = t * self.f_values[k] + (1 - t) * f_extra
        self.nsamples[k] += 1

        self.kopt = np.argmin(self.f_values[:self.npt()])  # make sure kopt is always the best value we have
        return

    def add_new_point(self, x, f, derivs):
        if self.npt() >= (self.n() + 1) * (self.n() + 2) // 2:
            return False  # cannot add more points

        self.points = np.append(self.points, x.reshape((1, self.n())), axis=0)  # append row to xpt
        self.f_values = np.append(self.f_values, f)  # append entry to f_values
        self.fderivs = np.append(self.fderivs, derivs)
        self.nsamples = np.append(self.nsamples, 1)  # add new sample number
        self.num_pts += 1  # make sure npt is updated

        if f < self.fopt():
            self.kopt = self.npt() - 1

        self.lu_current = False
        return True

    def shift_base(self, xbase_shift):
        # Shifting xbase -> xbase + xbase_shift
        for k in range(self.npt()):
            self.points[k, :] = self.points[k, :] - xbase_shift
        self.xbase += xbase_shift
        self.sl = self.sl - xbase_shift
        self.su = self.su - xbase_shift
        self.factorisation_current = False

        # Update model (always centred on xbase)
        Hx = self.model_hess.dot(xbase_shift)
        self.model_const += np.dot(self.model_grad + 0.5*Hx, xbase_shift)
        self.model_grad += Hx
        return

    def save_point(self, x, f, nsamples, derivs, x_in_abs_coords=True):
        if self.fsave is None or f <= self.fsave:
            self.xsave = x.copy() if x_in_abs_coords else self.as_absolute_coordinates(x)
            self.fsave = f
            self.derivssave = derivs
            self.gradsave = self.model_grad.copy()
            self.hesssave = self.model_hess.copy()
            self.nsamples_save = nsamples
            return True
        else:
            return False  # this value is worse than what we have already - didn't save

    def get_final_results(self):
        # Return x and fval for optimal point (either from xsave+fsave or kopt)
        if self.fsave is None or self.fopt() <= self.fsave:  # optimal has changed since xsave+fsave were last set
            g, hess = self.build_full_model()  # model based at xopt
            return self.xopt(abs_coordinates=True).copy(), self.fopt(), self.derivsopt(), g, hess, self.nsamples[self.kopt]
        else:
            return self.xsave, self.fsave, self.derivssave, self.gradsave, self.hesssave, self.nsamples_save

    def min_objective_value(self):
        # Get termination criterion for f small: f <= abs_tol
        return self.abs_tol

    def model_value(self, d, d_based_at_xopt=True, with_const_term=False):
        # Model is always centred around xbase
        const = self.model_const if with_const_term else 0.0
        d_to_use = d + self.xopt() if d_based_at_xopt else d
        Hd = self.model_hess.dot(d_to_use)
        return const + np.dot(self.model_grad + 0.5 * Hd, d_to_use)

    def interpolation_matrix(self):
        Y = self.xpt_directions(include_kopt=False).T
        if self.precondition:
            approx_delta = sqrt(np.max(self.distances_to_xopt()))  # largest distance to xopt ~ delta
        else:
            approx_delta = 1.0
        return build_interpolation_matrix(self,Y, approx_delta=approx_delta)

    def factorise_interp_matrix(self):
        if not self.factorisation_current:
            A, self.left_scaling, self.right_scaling, Areg, left_scaling_reg, right_scaling_reg = self.interpolation_matrix()
            self.lu, self.piv = LA.lu_factor(A)
            self.factorisation_current = True
        return

    def solve_system(self, rhs):
        # solve system with interpolation (LU factorization)
        # To do preconditioning below, we will need to scale each column of A elementwise by the entries of some vector
        col_scale = lambda A, scale: (A.T * scale).T  # Uses the trick that A*x scales the 0th column of A by x[0], etc.
        if self.factorisation_current:
            # A(preconditioned) = diag(left_scaling) * A(original) * diag(right_scaling)
            # Solve A(original)\rhs
            return col_scale(LA.lu_solve((self.lu, self.piv), col_scale(rhs, self.left_scaling)), self.right_scaling)
        else:
            if self.do_logging:
                logging.warning("model.solve_system not using factorisation")
            A, left_scaling, right_scaling, Areg, left_scaling_reg, right_scaling_reg = self.interpolation_matrix()
            return col_scale(LA.solve(A, col_scale(rhs, left_scaling)), right_scaling)  
    
    def solve_system_with_regression_scaled(self, rhs_reg, orig_system = False):
        # solve system with least squares regression
            # same scalarization as in solve_system
        # evaluate interpolation matrix and scalarization
        A, left_scaling, right_scaling, Areg, left_scaling_reg, right_scaling_reg = self.interpolation_matrix()
        # scale rhs according to diag(left_scaling_reg) * rhs
        rhs_scaled = np.dot(np.diag(left_scaling_reg),rhs_reg)
        
        # solve system with least squares
        lstsq_sol = np.linalg.lstsq(Areg, rhs_scaled, rcond=None)
        # get solution
        sol_scaled = lstsq_sol[0]
        # de-scale solution
            # solution was LARR^(-1)x = L*rhs, sol_scaled = R^(-1)x, we want to find x = sol = RR^(-1)x = R * sol_scaled
        sol = np.dot(np.diag(right_scaling_reg),sol_scaled)
        return sol, lstsq_sol
    
    
    def deriv_weights(self,Y):
        # optional weights for derivative lines
        weight_type = 'linear'
        weight_type = 'exp'
        weight_type = 'none'
        
        dist_list = np.zeros(len(Y.T))
        for i in range(len(dist_list)):
            y = Y.T[i]
            dist_list[i] = LA.norm(y)
        dist_max = max(dist_list)
        
        if weight_type == 'linear':
            scal_fac = 0.8 #1 #0.8 #max weight = 1, min weight = 1-scal_fac
            weight_list = np.zeros(len(Y.T))
            for i in range(len(dist_list)):
                weight_list[i] = 1 - scal_fac * (dist_list[i]/dist_max)
        
        elif weight_type == 'exp':
            scal_fac = 5
            exp_list = np.zeros(len(Y.T))
            for i in range(len(dist_list)):
                exp_list[i] = scal_fac * (1 - (dist_list[i]/dist_max))
                
            weight_list = np.zeros(len(Y.T))
            for i in range(len(dist_list)):
                weight_list[i] = np.exp(exp_list[i]) * (1/np.exp(scal_fac))
        
        else:
            # activating this line indicates: no weights
            weight_list = np.ones(len(Y.T))

        return weight_list
    
    
    def fct_weights(self,Y):
        # optional weights for function lines
        weight_type = 'linear'
        weight_type = 'exp'
        weight_type = 'none'
        
        # optional weights for function lines
        dist_list = np.zeros(len(Y.T))
        for i in range(len(dist_list)):
            y = Y.T[i]
            dist_list[i] = LA.norm(y)
        dist_max = max(dist_list)
        
        if weight_type == 'linear':
            scal_fac = 0.8 #1 #0.8 #max weight = 1, min weight = 1-scal_fac
            weight_list = np.zeros(len(Y.T))
            for i in range(len(dist_list)):
                weight_list[i] = 1 - scal_fac * (dist_list[i]/dist_max)
        
        elif weight_type == 'exp':
            scal_fac = 5
            exp_list = np.zeros(len(Y.T))
            for i in range(len(dist_list)):
                exp_list[i] = scal_fac * (1 - (dist_list[i]/dist_max))
                
            weight_list = np.zeros(len(Y.T))
            for i in range(len(dist_list)):
                weight_list[i] = np.exp(exp_list[i]) * (1/np.exp(scal_fac))
        
        else:
            # activating this line indicates: no weights
            weight_list = np.ones(len(Y.T))
            
        return weight_list
    

    
    def interpolate_model(self, verbose=False, min_chg_hess=True, get_norm_model_chg=False):
        # build model by interpolation or regression
        if verbose:
            A, left_scaling, right_scaling, Areg = self.interpolation_matrix()
            
            interp_cond_num = np.linalg.cond(A)  # scipy.linalg does not have condition number!
        else:
            interp_cond_num = 0.0
        #self.factorise_interp_matrix()
        
        fval_row_idx = np.delete(np.arange(self.npt()), self.kopt)  # indices of all rows except kopt
        fderivs_wo_opt = self.fderivs[fval_row_idx] # derivative values of all data points except xopt
        
        # set weights with weight function using all data points except xopt
        Y_wo_opt = self.xpt_directions(include_kopt=False).T
        weight_list = self.deriv_weights(Y_wo_opt)
        weight_list_fct = self.fct_weights(Y_wo_opt)


        # VERSION 1: Hermite least squares
        # Solve linear system of equations (LSE):
            # Q(y_i) = f(y_i), for all y_i in Y
            # dQ/dy_j(y_i) = df/dy_j(y_i), for all y_i in Y and known derivative directions j
        if self.version == 1:

            # build right hand side (rhs) for classic interpolation with function values
            # ... f(yopt) + g^T y +0.5 y^T H y = f(y) <=> g^T Y +0.5 y^T H y = f(y) - f(yopt) =: rhs
            rhs = self.f_values[fval_row_idx] - self.fopt()
            rhs_reg = rhs.copy()
            
            # consider weights for function lines
            for i in range(self.npt()-1):
                rhs_reg[i] = weight_list_fct[i] * rhs_reg[i]
            
            # add derivative lines
                # ... g + Hy = f'(y)
            # for each point (except xopt)
            for j in range(self.npt()-1):
                rhs_deriv_j_all = weight_list[j] * fderivs_wo_opt[j]  # all partial derivatives for j-th point (considering weights)
                rhs_deriv_j = rhs_deriv_j_all[self.known_derivs] # only known partial derivatives
                rhs_reg = np.concatenate((rhs_reg,rhs_deriv_j),axis=0) # add to existing rhs
            
            # derivative lines for current xopt
            rhs_deriv_opt_all = self.fderivs[self.kopt] # all partial derivatives for xopt (weight = 1)
            rhs_deriv_opt = rhs_deriv_opt_all[self.known_derivs] # only known partial derivatives
            rhs_reg = np.concatenate((rhs_reg,rhs_deriv_opt),axis=0) # add to existing rhs (at the end)
                
            # solve system with regression
            try:
                coeffs, lstsq_sol = self.solve_system_with_regression_scaled(rhs_reg)
            except LA.LinAlgError:
                return False, interp_cond_num, None, None, None  # flag error
            except ValueError:
                return False, interp_cond_num, None, None, None  # flag error (e.g. inf or NaN encountered)
    
            if not np.all(np.isfinite(coeffs)):  # another check for inf or NaN
                return False, interp_cond_num, None, None, None  # flag error
    
            # Old gradient and Hessian (save so can compute changes later)
            if verbose or get_norm_model_chg:
                old_model_grad = self.model_grad.copy()
                old_model_hess = self.model_hess.copy()
            else:
                old_model_grad = None
                old_model_hess = None
    
            # Build model from coefficients
            self.model_const = self.fopt()  # true in all cases
            self.model_grad = coeffs[:self.n()]
            self.model_hess = build_symmetric_matrix_from_vector(self.n(), coeffs[self.n():])  # rest of coeffs are upper triangular part of Hess

        # Base model at xbase, not xopt (note negative signs)
        xopt = self.xopt()
        Hx = self.model_hess.dot(xopt)
        self.model_const += np.dot(-self.model_grad + 0.5*Hx, xopt)
        self.model_grad += -Hx

        interp_error = 0.0
        norm_chg_grad = 0.0
        norm_chg_hess = 0.0
        if verbose or get_norm_model_chg:
            norm_chg_grad = LA.norm(self.model_grad - old_model_grad)
            norm_chg_hess = LA.norm(self.model_hess - old_model_hess, ord='fro')
        if verbose:
            for k in range(self.npt()):
                f_pred = self.model_value(self.xpt(k), d_based_at_xopt=False, with_const_term=True)
                interp_error += self.nsamples[k] * (self.f_values[k] - f_pred)**2
            interp_error = sqrt(interp_error)
            print('error',interp_error)  

        return True, interp_cond_num, norm_chg_grad, norm_chg_hess, interp_error  # flag ok


    def build_full_model(self):
        # Make model centred around xopt
        g = self.model_grad + self.model_hess.dot(self.xopt())
        return g, self.model_hess


    def lagrange_polynomial(self, k, factorise_first=True):
        # calculate Lagrange polynomial (for maintaining lambda-poisedness)
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        if factorise_first:
            self.factorise_interp_matrix()

        if k < self.kopt:
            k_row_idx = k
        elif k > self.kopt:
            k_row_idx = k-1
        else:
            k_row_idx = -1  # flag k==kopt
        
        # Define system matrix for Lagrange polynomials
        A, left_scaling, right_scaling, Areg, left_scaling_reg, right_scaling_reg = self.interpolation_matrix()
        if self.version == 1:
            # use regression matriux/system for Lagrange polynomials
            Alag = Areg
        Arows, Acolumns = np.shape((Alag))
        
        if self.npt() == self.n() + 1:
            if k_row_idx >= 0:
                rhs_lag = np.zeros((Arows))
                rhs_lag[k_row_idx] = 1.0
            else:
                rhs_lag = -np.ones((Arows))
        elif self.npt() == (self.n() + 1) * (self.n() + 2) // 2:
            if k_row_idx >= 0:
                rhs_lag = np.zeros((Arows))
                rhs_lag[k_row_idx] = 1.0
            else:
                rhs_lag = -np.ones((Arows))
        else:
            rhs_lag = np.zeros((Arows))
            if k_row_idx >= 0:
                rhs_lag[k_row_idx] = 1.0
            else:
                rhs_lag[:self.npt() - 1] = -1.0  # rest of entries are zero    
        
        # Solve system
        if self.version == 1 or self.version == 4:
            # using regression
            coeffs, lstsq_sol = self.solve_system_with_regression_scaled(rhs_lag)
        elif self.version == 2 or self.version == 3:
            # using interpolation
            coeffs = self.solve_system(rhs_lag)

        # Build polynomial from coefficients
        if self.version == 1:
            c = 1.0 if k==self.kopt else 0.0  # true in all cases
            g = coeffs[:self.n()]
            H = build_symmetric_matrix_from_vector(self.n(), coeffs[self.n():])  # rest of coeffs are upper triangular part of Hess
        elif self.version == 2 or self.version == 3 or self.version == 4:
            c = 1.0 if k==self.kopt else 0.0  # true in all cases
            if self.npt() == self.n() + 1:
                g = coeffs.copy()
                H = np.zeros((self.n(), self.n()))
            elif self.npt() == (self.n() + 1) * (self.n() + 2) // 2:
                g = coeffs[:self.n()]
                H = build_symmetric_matrix_from_vector(self.n(), coeffs[self.n():])  # rest of coeffs are upper triangular part of Hess
            else:
                g = coeffs[self.npt() - 1:]  # last n values
                fval_row_idx = np.delete(np.arange(self.npt()), self.kopt)  # indices of all rows except kopt
                H = np.zeros((self.n(), self.n()))
                for i in range(self.npt() - 1):
                    dx = self.xpt(fval_row_idx[i]) - self.xopt()
                    H += coeffs[i] * np.outer(dx, dx)
        
        # (c, g, hess) currently based around xopt
        return c, g, H


    def poisedness_constant(self, delta, xbase=None, xbase_in_abs_coords=True):
        # Calculate the poisedness constant of the current interpolation set in B(xbase, delta)
        # if xbase is None, use self.xopt()
        overall_max = None
        if xbase is None:
            xbase = self.xopt()
        elif xbase_in_abs_coords:
            xbase = xbase - self.xbase  # shift to correct position
        for k in range(self.npt()):
            c, g, H = self.lagrange_polynomial(k, factorise_first=True)  # based at self.xopt()
            # Switch base of poly from xopt to xbase, as required by trsbox_geometry
            base_chg = self.xopt() - xbase
            Hx = H.dot(base_chg)
            c += np.dot(-g + 0.5 * Hx, base_chg)
            g += -Hx
            xmax = trsbox_geometry(xbase, c, g, H, self.sl, self.su, delta)
            lmax = abs(c + model_value(g, H, xmax-xbase))  # evaluate Lagrange poly
            if overall_max is None or lmax > overall_max:
                overall_max = lmax
        return overall_max


def build_interpolation_matrix(self,Y, approx_delta=1.0):
    # Build system matrix of interpolation or regression problem
    # Y has columns Y[:,t] = yt - xk
    n, p = Y.shape  # p = npt-1
    assert n + 1 <= p + 1 <= (n + 1) * (n + 2) // 2, "npt must be in range [n+1, (n+1)(n+2)/2]"
    if self.version == 3 or self.version == 4:
        assert n + 1 < p + 1 < (n + 1) * (n + 2) // 2, "npt must be in range (n+1, (n+1)(n+2)/2). \n For fully linear or quadratic cases Version 3 is not allowed."
    if self.version == 1 or self.version == 2:
        assert (p+1) * (1+len(self.known_derivs)) >= ((n + 1) * (n + 2)) // 2, "Transformed system is underdetermined"
    
    fval_row_idx = np.delete(np.arange(self.npt()), self.kopt)  # indices of all rows except kopt

    # set weights with weight function using all data points except xopt
    weight_list = self.deriv_weights(Y)
    weight_list_fct = self.fct_weights(Y)
    
    # Original A from PyBOBYQA (for Lagrange polynomials in some versions)
    if p == n:  # linear models
        A = Y.T / approx_delta
        left_scaling = np.ones((n,))  # no left scaling
        right_scaling = np.ones((n,)) / approx_delta
    elif p + 1 == (n+1)*(n+2)//2:  # fully quadratic models
        A = np.zeros((p, p))
        A[:,:n] = Y.T / approx_delta
        for i in range(p):
            A[i, n:] = to_upper_triangular_vector(np.outer(Y[:,i], Y[:,i]) - 0.5*np.diag(np.square(Y[:,i]))) / (approx_delta**2)
        left_scaling = np.ones((p,))  # no left scaling
        right_scaling = np.ones((p,))
        right_scaling[:n] = 1.0 / approx_delta
        right_scaling[n:] = 1.0 / (approx_delta**2)
    else:  # underdetermined quadratic models
        A = np.zeros((p + n, p + n))
        for i in range(p):
            for j in range(p):
                A[i,j] = 0.5*np.dot(Y[:,i], Y[:,j])**2 / (approx_delta**4)
        A[:p,p:] = Y.T / approx_delta
        A[p:,:p] = Y / approx_delta
        left_scaling = np.ones((p+n,))
        right_scaling = np.ones((p + n,))
        left_scaling[:p] = 1.0 / (approx_delta**2)
        left_scaling[p:] = approx_delta
        right_scaling[:p] = 1.0 / (approx_delta**2)
        right_scaling[p:] = approx_delta
    
    # VERSION 1: A for Hermite least squares approach
    # Modified A --> Areg for regression with derivative information
    # Add derivative lines to fully or underdetermined system
    if self.version == 1:
        # Set system matrix for function value rows (not for derivatives)
        A_fct = np.zeros((p,((n+1)*(n+2)//2)-1))
        # linear columns
        A_fct[:,:n] = Y.T / approx_delta
        # quadratic columns
        for i in range(p):
            A_fct[i, n:] = to_upper_triangular_vector(np.outer(Y[:,i], Y[:,i]) - 0.5*np.diag(np.square(Y[:,i]))) / (approx_delta**2)
        Areg = A_fct.copy()
        # consider weights (optional)
        for i in range(p):
            Areg[i] = weight_list_fct[i] * Areg[i]
        
        # Define left and right scaling vectors for modified system
        # ... diag(ls_bobyqa) A_reg diag(rs_bobyqa)
        ls_bobyqa = np.ones((p,))  # no left scaling
        rs_bobyqa = np.ones((((n+1)*(n+2)//2)-1,))
        rs_bobyqa[:n] = 1.0 / approx_delta
        rs_bobyqa[n:] = 1.0 / (approx_delta**2)
        
        # Set system matrix for derivative rows
        # for deriv_comp-th derivative direction
        for i in range(len(self.known_derivs)):
            deriv_comp = self.known_derivs[i] # direction of this partial derivative
            
            # initialize matrix with derivatives of all points in Y (except yopt) with respect to the direction deriv_comp
            A_deriv_i = np.zeros((p,((n+1)*(n+2)//2)-1))
            A_deriv_i[:,deriv_comp] = 1  # linear columns
            
            # for j-th interpolation point
            for j in range(p):
                deriv_ij = np.zeros((n,n))
                deriv_ij[deriv_comp,:] = Y.T[j,:] / approx_delta
                A_line_i = to_upper_triangular_vector_MF(deriv_ij) # quadratic columns
                A_deriv_i[j,n:] = A_line_i # all columns
            
            # A_deriv contains all (available) derivative information for all points in Y (except yopt)
            if i == 0:
                A_deriv = A_deriv_i
            else:
                A_deriv = np.concatenate((A_deriv,A_deriv_i),axis=0)
        
        # order according to points and not derivative directions
        for points in range(p):
            for derivs in range(len(self.known_derivs)):
                nextline = weight_list[points] * np.array([A_deriv[(derivs)*p + points]])
                Areg = np.concatenate((Areg,nextline),axis=0)
        
        # Set Y including yopt
        Y_incl_opt = self.xpt_directions(include_kopt=True).T

        # Add derivative information for current optimal solution 
        popt = 1 # only one optimal solution --> loop over popt=1
        for i in range(len(self.known_derivs)):
            deriv_comp = self.known_derivs[i]
            A_deriv_i = np.zeros((popt,((n+1)*(n+2)//2)-1))
            A_deriv_i[:,deriv_comp] = 1
            
            # only for the one optimal interpolation point
            for j in range(popt):
                deriv_ij = np.zeros((n,n))
                deriv_ij[deriv_comp,:] = Y_incl_opt.T[self.kopt,:] / approx_delta
                A_line_i = to_upper_triangular_vector_MF(deriv_ij)
                A_deriv_i[j,n:] = A_line_i
        
            Areg = np.concatenate((Areg,A_deriv_i),axis=0)
        
        # Update left scaling for Areg
        nrows_Areg, ncol_Areg = np.shape(Areg)
        #ls_bobyqa = np.ones((nrows_Areg,))
        ls_bobyqa = np.concatenate((ls_bobyqa,approx_delta*np.ones((nrows_Areg-p,))))
            
    return A, left_scaling, right_scaling, Areg, ls_bobyqa, rs_bobyqa


def build_symmetric_matrix_from_vector(n, entries):
    assert entries.shape == (n*(n+1)//2,), "Entries vector has wrong size, got %g, expect %g (for n=%g)" % (len(entries), n*(n+1)//2, n)
    A = np.zeros((n, n))
    ih = -1
    for j in range(n):  # j = 0, ..., n-1
        for i in range(j + 1):  # i = 0, ..., j
            ih += 1
            A[i, j] = entries[ih]
            A[j, i] = entries[ih]
    return A


