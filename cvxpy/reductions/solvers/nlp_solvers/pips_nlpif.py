"""
Copyright 2025, the CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import (
    Equality,
    Inequality,
    NonPos,
)
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import NLPsolver
from cvxpy.reductions.utilities import (
    lower_equality,
    lower_ineq_to_nonneg,
    nonpos2nonneg,
)


class PIPS(NLPsolver):
    """
    NLP interface for the PIPS (Python Interior Point Solver) from PYPOWER
    """
    
    # Map between PIPS eflag and CVXPY status
    STATUS_MAP = {
        True: s.OPTIMAL,      # Converged
        False: s.USER_LIMIT,  # Did not converge (max iterations)
        -1: s.SOLVER_ERROR,   # Numerically failed
    }

    def name(self):
        """The name of solver."""
        return 'PIPS'

    def import_solver(self):
        """Imports the solver."""
        from pypower.pips import pips  # noqa F401

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
        attr = {}
        
        # Map eflag to CVXPY status
        if solution['eflag'] == 1:
            status = s.OPTIMAL
        elif solution['eflag'] == 0:
            status = s.USER_LIMIT
        else:  # -1
            status = s.SOLVER_ERROR
            
        attr[s.NUM_ITERS] = solution['output']['iterations']
        
        # Extract additional statistics from history if available
        if 'hist' in solution['output'] and len(solution['output']['hist']) > 0:
            final_hist = solution['output']['hist'][-1]
            attr['feascond'] = final_hist['feascond']
            attr['gradcond'] = final_hist['gradcond']
            attr['compcond'] = final_hist['compcond']
            attr['costcond'] = final_hist['costcond']
    
        if status in s.SOLUTION_PRESENT:
            primal_val = solution['f']
            opt_val = primal_val + inverse_data.offset
            primal_vars = {}
            x_opt = solution['x']
            
            for id, offset in inverse_data.var_offsets.items():
                shape = inverse_data.var_shapes[id]
                size = np.prod(shape, dtype=int)
                primal_vars[id] = np.reshape(x_opt[offset:offset+size], shape, order='F')
                
            # Extract dual variables if available
            dual_vars = {}
            if 'lmbda' in solution:
                lmbda = solution['lmbda']
                # Map PIPS dual variables to CVXPY constraints
                # This would require more sophisticated mapping based on constraint types
                
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """
        Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data used by the solver.
        warm_start : bool
            Not used (PIPS doesn't support warm start).
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.
        solver_cache: None
            None

        Returns
        -------
        dict
            Solution dictionary from PIPS
        """
        from pypower.pips import pips
        
        # Setup bounds and initial point
        bounds = self.Bounds(data["problem"])
        x0 = self.construct_initial_point(bounds)
        
        # Create oracles for function evaluations
        oracles = self.Oracles(bounds.new_problem, x0, bounds)
        
        # Setup linear constraints if any
        A, l, u = self.extract_linear_constraints(bounds)
        
        # Setup default options
        default_options = {
            'feastol': 1e-6,
            'gradtol': 1e-6,
            'comptol': 1e-6,
            'costtol': 1e-6,
            'max_it': 150,
            'step_control': False,
            'cost_mult': 1.0,
            'verbose': 1 if verbose else 0
        }
        
        # Update with user options
        if solver_opts:
            default_options.update(solver_opts)
            
        # Call PIPS solver
        solution = pips(
            f_fcn=oracles.f_fcn,
            x0=x0,
            A=A,
            l=l,
            u=u,
            xmin=bounds.lb,
            xmax=bounds.ub,
            gh_fcn=oracles.gh_fcn if oracles.has_nonlinear else None,
            hess_fcn=oracles.hess_fcn,
            opt=default_options
        )
        
        return solution

    def extract_linear_constraints(self, bounds):
        """Extract linear constraints in PIPS format (l <= Ax <= u)"""
        # For now, return None to indicate no linear constraints
        # A full implementation would extract linear constraints from the problem
        return None, None, None

    def construct_initial_point(self, bounds):
        """Construct initial point based on variable bounds and values"""
        initial_values = []
        offset = 0
        lbs = bounds.lb 
        ubs = bounds.ub
        
        for var in bounds.main_var:
            if var.value is not None:
                initial_values.append(np.atleast_1d(var.value).flatten(order='F'))
            else:
                # Initialize based on bounds
                lb = lbs[offset:offset + var.size]
                ub = ubs[offset:offset + var.size]

                lb_finite = np.isfinite(lb)
                ub_finite = np.isfinite(ub)

                lb0 = np.where(lb_finite, lb, 0.0)
                ub0 = np.where(ub_finite, ub, 0.0)

                # Midpoint if both finite, one from bound if only one finite, zero if none
                init = (lb_finite * ub_finite * 0.5 * (lb0 + ub0) +
                       lb_finite * (~ub_finite) * (lb0 + 1.0) +
                       (~lb_finite) * ub_finite * (ub0 - 1.0))

                initial_values.append(init)
            
            offset += var.size
            
        x0 = np.concatenate(initial_values, axis=0)
        return x0

    class Oracles():
        """Oracle functions for PIPS solver"""
        
        def __init__(self, problem, initial_point, bounds):
            self.problem = problem
            self.bounds = bounds
            self.main_var = []
            for var in self.problem.variables():
                self.main_var.append(var)
            
            # Check if we have nonlinear constraints
            self.has_nonlinear = len(problem.constraints) > 0
            
            # Pre-allocate arrays for efficiency
            self.grad_obj = np.zeros(initial_point.size)
            self.hess_lag_values = None
            self.hess_lag_structure = None
            
        def set_variable_value(self, x):
            """Set CVXPY variable values from numpy array"""
            offset = 0
            for var in self.main_var:
                size = var.size
                var.value = x[offset:offset+size].reshape(var.shape, order='F')
                offset += size
        
        def f_fcn(self, x, return_hessian=False):
            """
            Evaluate objective function and its derivatives.
            
            Returns:
                f: scalar objective value
                df: gradient vector
                d2f: Hessian matrix (only if return_hessian=True)
            """
            self.set_variable_value(x)
            
            # Objective value
            f = self.problem.objective.args[0].value
            
            # Gradient
            df = np.zeros(x.size)
            grad_offset = 0
            grad_dict = self.problem.objective.expr.grad
            
            for var in self.main_var:
                size = var.size
                if var in grad_dict:
                    array = grad_dict[var]
                    if sp.issparse(array):
                        array = array.toarray().flatten(order='F')
                    df[grad_offset:grad_offset+size] = array
                grad_offset += size
            
            if return_hessian:
                # Compute Hessian of objective only
                hess_dict = self.problem.objective.expr.hess_vec(np.array([1.0]))
                d2f = self._build_hessian_matrix(hess_dict, x.size)
                return f, df, d2f
            else:
                return f, df
        
        def gh_fcn(self, x):
            """
            Evaluate nonlinear constraints and their gradients.
            
            Returns:
                h: inequality constraints (h <= 0)
                g: equality constraints (g = 0)
                dh: Jacobian of inequalities (sparse)
                dg: Jacobian of equalities (sparse)
            """
            self.set_variable_value(x)
            
            g_values = []  # Equality constraints
            h_values = []  # Inequality constraints
            dg_rows, dg_cols, dg_data = [], [], []
            dh_rows, dh_cols, dh_data = [], [], []
            
            g_row_offset = 0
            h_row_offset = 0
            
            for constraint in self.problem.constraints:
                # Get constraint value
                constr_val = np.asarray(constraint.args[0].value).flatten(order='F')
                
                # Determine constraint type from bounds
                constr_idx = self.problem.constraints.index(constraint)
                start_idx = sum(c.size for c in self.problem.constraints[:constr_idx])
                end_idx = start_idx + constraint.size
                
                cl = self.bounds.cl[start_idx:end_idx]
                cu = self.bounds.cu[start_idx:end_idx]
                
                # Check if equality (cl == cu == 0) or inequality
                is_equality = np.allclose(cl, cu) and np.allclose(cl, 0)
                
                if is_equality:
                    g_values.append(constr_val)
                    # Build gradient
                    col_offset = 0
                    grad_dict = constraint.expr.grad
                    for var in self.main_var:
                        if var in grad_dict:
                            jacobian = grad_dict[var].T
                            if sp.issparse(jacobian):
                                jacobian = jacobian.tocoo()
                                dg_rows.extend(jacobian.row + g_row_offset)
                                dg_cols.extend(jacobian.col + col_offset)
                                dg_data.extend(jacobian.data)
                            else:
                                # Dense jacobian
                                jac_flat = jacobian.flatten(order='F')
                                for i, val in enumerate(jac_flat):
                                    if val != 0:
                                        dg_rows.append(g_row_offset + i // var.size)
                                        dg_cols.append(col_offset + i % var.size)
                                        dg_data.append(val)
                        col_offset += var.size
                    g_row_offset += constraint.size
                else:
                    h_values.append(constr_val)
                    # Build gradient
                    col_offset = 0
                    grad_dict = constraint.expr.grad
                    for var in self.main_var:
                        if var in grad_dict:
                            jacobian = grad_dict[var].T
                            if sp.issparse(jacobian):
                                jacobian = jacobian.tocoo()
                                dh_rows.extend(jacobian.row + h_row_offset)
                                dh_cols.extend(jacobian.col + col_offset)
                                dh_data.extend(jacobian.data)
                            else:
                                # Dense jacobian
                                jac_flat = jacobian.flatten(order='F')
                                for i, val in enumerate(jac_flat):
                                    if val != 0:
                                        dh_rows.append(h_row_offset + i // var.size)
                                        dh_cols.append(col_offset + i % var.size)
                                        dh_data.append(val)
                        col_offset += var.size
                    h_row_offset += constraint.size
            
            # Concatenate constraint values
            g = np.concatenate(g_values) if g_values else np.array([])
            h = np.concatenate(h_values) if h_values else np.array([])
            
            # Build sparse Jacobians
            n_vars = sum(var.size for var in self.main_var)
            
            if g.size > 0:
                dg = sp.csr_matrix((dg_data, (dg_rows, dg_cols)), 
                                  shape=(g.size, n_vars))
            else:
                dg = None
                
            if h.size > 0:
                dh = sp.csr_matrix((dh_data, (dh_rows, dh_cols)),
                                  shape=(h.size, n_vars))
            else:
                dh = None
            
            return h, g, dh, dg
        
        def hess_fcn(self, x, lmbda, cost_mult=1.0):
            """
            Compute Hessian of the Lagrangian.
            
            Parameters:
                x: current point
                lmbda: dict with keys 'eqnonlin' and 'ineqnonlin'
                cost_mult: scaling factor for objective
                
            Returns:
                Lxx: sparse Hessian of Lagrangian
            """
            self.set_variable_value(x)
            n_vars = x.size
            
            # Initialize Hessian with objective contribution
            hess_dict = self.problem.objective.expr.hess_vec(np.array([cost_mult]))
            hess_lagrangian = self._build_hessian_matrix(hess_dict, n_vars)
            
            # Add constraint contributions
            if self.has_nonlinear:
                # Map constraints to equality/inequality based on bounds
                eq_idx = 0
                ineq_idx = 0
                
                for constraint in self.problem.constraints:
                    constr_idx = self.problem.constraints.index(constraint)
                    start_idx = sum(c.size for c in self.problem.constraints[:constr_idx])
                    end_idx = start_idx + constraint.size
                    
                    cl = self.bounds.cl[start_idx:end_idx]
                    cu = self.bounds.cu[start_idx:end_idx]
                    is_equality = np.allclose(cl, cu) and np.allclose(cl, 0)
                    
                    if is_equality and 'eqnonlin' in lmbda:
                        dual_vec = lmbda['eqnonlin'][eq_idx:eq_idx + constraint.size]
                        eq_idx += constraint.size
                    elif not is_equality and 'ineqnonlin' in lmbda:
                        dual_vec = lmbda['ineqnonlin'][ineq_idx:ineq_idx + constraint.size]
                        ineq_idx += constraint.size
                    else:
                        continue
                    
                    # Get constraint Hessian
                    constr_hess_dict = constraint.expr.hess_vec(dual_vec)
                    constr_hess = self._build_hessian_matrix(constr_hess_dict, n_vars)
                    hess_lagrangian += constr_hess
            
            return sp.csr_matrix(hess_lagrangian)
        
        def _build_hessian_matrix(self, hess_dict, n_vars):
            """Build full Hessian matrix from dictionary of blocks"""
            hess_matrix = np.zeros((n_vars, n_vars))
            row_offset = 0
            
            for var1 in self.main_var:
                col_offset = 0
                for var2 in self.main_var:
                    if (var1, var2) in hess_dict:
                        var_hess = hess_dict[(var1, var2)]
                        if sp.issparse(var_hess):
                            var_hess = var_hess.toarray()
                        hess_matrix[row_offset:row_offset+var1.size,
                                  col_offset:col_offset+var2.size] = var_hess
                    col_offset += var2.size
                row_offset += var1.size
                
            return hess_matrix

    class Bounds():
        """Extract and manage variable and constraint bounds"""
        
        def __init__(self, problem):
            self.problem = problem
            self.main_var = problem.variables()
            self.get_constraint_bounds()
            self.get_variable_bounds()

        def get_constraint_bounds(self):
            """Normalize constraints and extract bounds"""
            lower = []
            upper = []
            new_constr = []
            
            for constraint in self.problem.constraints:
                if isinstance(constraint, Equality):
                    lower.extend([0.0] * constraint.size)
                    upper.extend([0.0] * constraint.size)
                    new_constr.append(lower_equality(constraint))
                elif isinstance(constraint, Inequality):
                    lower.extend([0.0] * constraint.size)
                    upper.extend([np.inf] * constraint.size)
                    new_constr.append(lower_ineq_to_nonneg(constraint))
                elif isinstance(constraint, NonPos):
                    lower.extend([0.0] * constraint.size)
                    upper.extend([np.inf] * constraint.size)
                    new_constr.append(nonpos2nonneg(constraint))
            
            lowered_con_problem = self.problem.copy([self.problem.objective, new_constr])
            self.new_problem = lowered_con_problem
            self.cl = np.array(lower)
            self.cu = np.array(upper)

        def get_variable_bounds(self):
            """Extract variable bounds"""
            var_lower = []
            var_upper = []
            
            for var in self.main_var:
                size = var.size
                if var.bounds:
                    lb = var.bounds[0].flatten(order='F')
                    ub = var.bounds[1].flatten(order='F')

                    if var.is_nonneg():
                        lb = np.maximum(lb, 0)
                    
                    if var.is_nonpos():
                        ub = np.minimum(ub, 0)
                    
                    var_lower.extend(lb)
                    var_upper.extend(ub)
                else:
                    # No bounds specified
                    if var.is_nonneg():
                        var_lower.extend([0.0] * size)
                    else:
                        var_lower.extend([-np.inf] * size)
                    
                    if var.is_nonpos():
                        var_upper.extend([0.0] * size)
                    else:
                        var_upper.extend([np.inf] * size)

            self.lb = np.array(var_lower)
            self.ub = np.array(var_upper)
