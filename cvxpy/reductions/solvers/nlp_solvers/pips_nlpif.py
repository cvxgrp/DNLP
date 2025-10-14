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

import cvxpy as cp
from cvxpy.constraints.zero import Zero
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims
from cvxpy.reductions.inverse_data import InverseData
import cvxpy.settings as s
from cvxpy.constraints import (
    Equality,
    Inequality,
    NonPos,
)
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import NLPsolver
from cvxpy.reductions.utilities import (
    ReducedMat,
    group_constraints,
    lower_equality,
    lower_ineq_to_nonneg,
    lower_ineq_to_nonpos,
)
from cvxpy.utilities.citations import CITATION_DICT
from cvxpy.utilities.coeff_extractor import CoeffExtractor


class PIPS(NLPsolver):
    """
    NLP interface for the PIPS solver
    """
    # Map between PIPS status and CVXPY status
    STATUS_MAP = {
        # Success cases
        True: s.OPTIMAL,                     # converged=True: first order optimality
        False: s.OPTIMAL_INACCURATE,         # converged=False: maximum iterations reached
        None: s.SOLVER_ERROR,                # converged=None: numerically failed
    }

    def name(self):
        """
        The name of solver.
        """
        return 'PIPS'

    def import_solver(self):
        """
        Imports the solver.
        """
        import pypower  # noqa F401

    def invert(self, solution, inverse_data):
        """
        Returns the solution to the original problem given the inverse_data.
        """
        attr = {}
        status = self.STATUS_MAP[solution['converged']]
        
        # Extract PIPS output information
        attr[s.NUM_ITERS] = solution['output']['iterations']
        
        # Store PIPS-specific information if needed
        if 'output' in solution and 'message' in solution['output']:
            attr[s.EXTRA_STATS] = {'message': solution['output']['message']}
        
        if status in s.SOLUTION_PRESENT:
            primal_val = solution['f']
            opt_val = primal_val + inverse_data.offset
            primal_vars = {}
            x_opt = solution['x']
            for id, offset in inverse_data.var_offsets.items():
                shape = inverse_data.var_shapes[id]
                size = np.prod(shape, dtype=int)
                primal_vars[id] = np.reshape(x_opt[offset:offset+size], shape, order='F')
            
            # Extract dual variables from PIPS lambda dictionary
            dual_vars = {}
            # if 'lmbda' in solution:
                # lmbda = solution['lmbda']
                # Map PIPS dual variables to CVXPY constraint IDs
                # This mapping depends on your specific constraint setup
                # Example mappings (adjust based on your inverse_data):
                # dual_vars[constraint_id] = lmbda['eqnonlin'] or lmbda['ineqnonlin'], etc.
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
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.
        solver_cache: None
            None

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        from pypower.pips import pips
        bounds = self.Bounds(data["problem"])
        x0 = self.construct_initial_point(bounds)
        A, lin_lower, lin_upper = bounds.A, bounds.l, bounds.u
        # Create oracles object
        oracles = self.Oracles(bounds.new_problem, x0,
                               bounds.nonlinear_eq, bounds.nonlinear_ineq)
        import pdb; pdb.set_trace()
        # Set options
        solution_nl = pips(
            f_fcn=oracles.f_fcn, x0=x0, A=A, l=lin_lower, u=lin_upper,
            xmin=bounds.lb, xmax=bounds.ub,
            #gh_fcn=oracles.gh_fcn,
            #hess_fcn=oracles.hess_fcn,
            opt=None)
        print(f"Nonlinear solution: x = {solution_nl['x']}")
        print(f"Objective value: {solution_nl['f']}")
        print(f"Iterations: {solution_nl['output']['iterations']}")
        # return solution
        return solution_nl

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["PIPS"]

    def construct_initial_point(self, bounds):
        initial_values = []
        offset = 0
        lbs = bounds.lb 
        ubs = bounds.ub
        for var in bounds.main_var:
            if var.value is not None:
                initial_values.append(np.atleast_1d(var.value).flatten(order='F'))
            else:
                # If no initial value is specified, look at the bounds.
                # If both lb and ub are specified, we initialize the
                # variables to be their midpoints. If only one of them 
                # is specified, we initialize the variable one unit 
                # from the bound. If none of them is specified, we 
                # initialize it to zero.
                lb = lbs[offset:offset + var.size]
                ub = ubs[offset:offset + var.size]

                lb_finite = np.isfinite(lb)
                ub_finite = np.isfinite(ub)

                # Replace infs with zero for arithmetic
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
        def __init__(self, problem, initial_point,
                     equality_constr, inequality_constr):
            self.problem = problem
            self.grad_obj = np.zeros(initial_point.size, dtype=np.float64)

            self.hess_lagrangian_coo = ([], [], [])
            self.jacobian_coo = ([], [], [])

            self.equality_constr = equality_constr
            self.inequality_constr = inequality_constr

            self.initial_point = initial_point
            
            self.main_var = []
            for var in self.problem.variables():
                self.main_var.append(var)

        def set_variable_value(self, x):
            offset = 0
            for var in self.main_var:
                size = var.size
                var.value = x[offset:offset+size].reshape(var.shape, order='F')
                offset += size

        def f_fcn(self, x, return_hessian=False):
            """Returns the objective and gradient of the objective."""
            if return_hessian:
                zeros = np.zeros((x.size, x.size))
                return self.objective(x), self.gradient(x), zeros
            return self.objective(x), self.gradient(x)
    
        def objective(self, x):
            """Returns the scalar value of the objective given x."""
            self.set_variable_value(x)
            obj_value = self.problem.objective.args[0].value
            return obj_value
        
        def gradient(self, x):
            """Returns the gradient of the objective with respect to x."""
            self.set_variable_value(x)
            # fill with zeros to reset from previous call
            self.grad_obj.fill(0)
            grad_offset = 0
            grad_dict = self.problem.objective.expr.grad
            for var in self.main_var:
                size = var.size
                if var in grad_dict:
                    array = grad_dict[var]
                    if sp.issparse(array):
                        array = array.toarray().flatten(order='F')
                    self.grad_obj[grad_offset:grad_offset+size] = array
                grad_offset += size
            return self.grad_obj

        def gh_fcn(self, x):
            """
            Returns the constraint values and jacobian
            of the equality and inequality constraints.
            """
            g, dg = (self.constraints(x, self.equality_constr),
                     self.jacobian(x, self.equality_constr))
            h, dh = (self.constraints(x, self.inequality_constr),
                     self.jacobian(x, self.inequality_constr))
            return h, g, dh, dg
        
        def constraints(self, x, constr_list):
            """Returns the constraint values."""
            self.set_variable_value(x)
            # Evaluate all constraints
            constraint_values = []
            for constraint in constr_list:
                constraint_values.append(np.asarray(constraint.args[0].value).flatten(order='F'))
            if not constraint_values:
                return np.array([])
            return np.concatenate(constraint_values)

        def jacobian(self, x, constr_list):
            self.set_variable_value(x)
            # compute jacobian of each constraint in constr_list
            if not constr_list:
                return None
            constr_offset = 0
            for constraint in constr_list:
                grad_dict = constraint.expr.jacobian()
                self.parse_jacobian_dict(grad_dict, constr_offset)
                constr_offset += constraint.size
            import pdb; pdb.set_trace()
            coo = sp.coo_matrix(
                (self.jacobian_coo[2], (self.jacobian_coo[0], self.jacobian_coo[1])),
                shape=(constr_offset, self.initial_point.size))
            return coo.tocsr()

        def parse_jacobian_dict(self, grad_dict, constr_offset):
            col_offset = 0
            for var in self.main_var:
                if var in grad_dict:
                    rows, cols, vals = grad_dict[var]
                    if not isinstance(rows, np.ndarray):
                        rows = np.array(rows)
                    if not isinstance(cols, np.ndarray):
                        cols = np.array(cols)

                    self.jacobian_coo[0].extend(rows + constr_offset)
                    self.jacobian_coo[1].extend(cols + col_offset)
                    self.jacobian_coo[2].extend(vals)

                col_offset += var.size

        def hess_fcn(self, x, duals, obj_factor):
            equality_constr = self.equality_constr
            inequality_constr = self.inequality_constr
            if not equality_constr and not inequality_constr:
                return None
            return self.hessian(x, duals, obj_factor, equality_constr, inequality_constr)

        def hessian(self, x, duals, obj_factor, equality_constr, inequality_constr):
            self.set_variable_value(x)
            
            # reset previous call
            self.hess_lagrangian_coo = ([], [], [])
            
            # compute hessian of objective times obj_factor
            obj_hess_dict = self.problem.objective.expr.hess_vec(np.array([obj_factor]))
            self.parse_hess_dict(obj_hess_dict)

            # compute hessian of constraints times duals
            constr_offset = 0
            for constraint in equality_constr:
                lmbda = duals['eqnonlin'][constr_offset:constr_offset + constraint.size]
                constraint_hess_dict = constraint.expr.hess_vec(lmbda)
                self.parse_hess_dict(constraint_hess_dict)
                constr_offset += constraint.size
            # reset constraint offset for inequality duals
            constr_offset = 0
            for constraint in inequality_constr:
                lmbda = duals['ineqnonlin'][constr_offset:constr_offset + constraint.size]
                constraint_hess_dict = constraint.expr.hess_vec(lmbda)
                self.parse_hess_dict(constraint_hess_dict)
                constr_offset += constraint.size
            # merge duplicate entries together
            self.sum_coo()
            coo = sp.coo_matrix(
                (self.hess_lagrangian_coo[2], 
                 (self.hess_lagrangian_coo[0], self.hess_lagrangian_coo[1])),
                shape=(self.initial_point.size, self.initial_point.size))
            return coo.tocsr()

        def parse_hess_dict(self, hess_dict):
            """
            Adds the contribution of blocks defined in hess_dict to the full
            hessian matrix
            """
            row_offset = 0
            for var1 in self.main_var:
                col_offset = 0
                for var2 in self.main_var:
                    if (var1, var2) in hess_dict:
                        rows, cols, vals = hess_dict[(var1, var2)]
                        if not isinstance(rows, np.ndarray):
                            rows = np.array(rows)
                        if not isinstance(cols, np.ndarray):
                            cols = np.array(cols)

                        self.hess_lagrangian_coo[0].extend(rows + row_offset)
                        self.hess_lagrangian_coo[1].extend(cols + col_offset)
                        self.hess_lagrangian_coo[2].extend(vals)

                    col_offset += var2.size
                row_offset += var1.size

        def sum_coo(self):
            shape = (self.initial_point.size, self.initial_point.size)
            rows, cols, vals = self.hess_lagrangian_coo
            coo = sp.coo_matrix((vals, (rows, cols)), shape=shape)
            coo.sum_duplicates()
            self.hess_lagrangian_coo = (coo.row, coo.col, coo.data)


    class Bounds():
        """
        This class stores the bounds on the constraints and variables
        and also reorders the constraints in the following way:
        affine constraints, nonlinear equality constraints,
        nonlinear inequality constraints.
        """
        def __init__(self, problem):
            self.problem = problem
            self.main_var = problem.variables()
            self.order_constraints()
            self.normalize_constraints()
            self.get_variable_bounds()
            self.get_linear_data()

        def order_constraints(self):
            """
            Orders the constraints in the following way:
            affine constraints, nonlinear equality constraints,
            nonlinear inequality constraints.
            """
            affine_constr = []
            nonlinear_eq = []
            nonlinear_ineq = []
            # separate affine and nonlinear constraints
            for constraint in self.problem.constraints:
                if constraint.expr.is_affine():
                    if isinstance(constraint, Equality):
                        affine_constr.append(lower_equality(constraint))
                    elif isinstance(constraint, Inequality):
                        affine_constr.append(lower_ineq_to_nonneg(constraint))
                else:
                    if isinstance(constraint, Equality):
                        nonlinear_eq.append(constraint)
                    elif isinstance(constraint, (Inequality, NonPos, NonNeg)):
                        nonlinear_ineq.append(constraint)

            self.affine_constr = affine_constr
            self.nonlinear_eq = nonlinear_eq
            self.nonlinear_ineq = nonlinear_ineq

        def normalize_constraints(self):
            """
            Normalizes the constraints to a standard form.
            Equalities are converted to g(x) = 0
            Inequalities are converted to h(x) <= 0
            """
            new_nonlinear_eq = []
            new_nonlinear_ineq = []
            for constraint in self.nonlinear_eq:
                new_nonlinear_eq.append(lower_equality(constraint))
            
            for constraint in self.nonlinear_ineq:
                if isinstance(constraint, Inequality):
                    new_nonlinear_ineq.append(lower_ineq_to_nonpos(constraint))
                elif isinstance(constraint, NonNeg):
                    new_nonlinear_ineq.append(NonPos(-constraint.expr,
                                                     constr_id=constraint.constr_id))
                elif isinstance(constraint, NonPos):
                    new_nonlinear_ineq.append(constraint)

            new_constr = self.affine_constr + new_nonlinear_eq + new_nonlinear_ineq
            lowered_con_problem = self.problem.copy([self.problem.objective, new_constr])
            self.new_problem = lowered_con_problem

        def get_variable_bounds(self):
            var_lower, var_upper = [], []
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
                    # No bounds specified, use infinite bounds or bounds
                    # set by the nonnegative or nonpositive attribute
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

        def get_linear_data(self):
            """
            Use CVXPY's get_problem_data to get the matrix A
            and bounds l, u for the affine constraints.
            """
            """
            linear_problem = cp.Problem(cp.Minimize(0),
                                         self.affine_constr)
            data, _, _ = linear_problem.get_problem_data(cp.SCS)
            A, b, cones = data['A'], data['b'], data['dims']
            self.A = A
            """
            inverse_data = InverseData(self.new_problem)
            extractor = CoeffExtractor(inverse_data, s.SCIPY_CANON_BACKEND)
            # Reorder constraints to Zero, NonNeg, SOC, PSD, EXP, PowCone3D
            constr_map = group_constraints(self.affine_constr)
            ordered_cons = constr_map[Zero] + constr_map[NonNeg]
            inverse_data.cons_id_map = {con.id: con.id for con in ordered_cons}
            inverse_data.constraints = ordered_cons
            # Batch expressions together, then split apart.
            expr_list = [arg for c in ordered_cons for arg in c.args]
            params_to_problem_data = extractor.affine(expr_list)
            var_size = 0
            for var in self.main_var:
                var_size += var.size
            reduced_A = ReducedMat(params_to_problem_data, var_size)
            reduced_A.cache(keep_zeros=False)
            A, b = reduced_A.get_matrix_from_tensor(None, with_offset=True)
            self.A = A
            b = np.atleast_1d(b)
            constr_map = group_constraints(ordered_cons)
            cones = ConeDims(constr_map)
            lower = -np.inf * np.ones(b.shape)
            upper = np.inf * np.ones(b.shape)
            import pdb; pdb.set_trace()
            # the standard form of SCS is Ax + s = b, s in K
            # in other words we have b - Ax in {0} x {R_+}
            if cones.zero > 0:
                lower[:cones.zero] = b[:cones.zero]
                upper[:cones.zero] = b[:cones.zero]
            # the remainder of the constraints are nonnegative
            # in other words b >= Ax >= -inf
            if cones.nonneg > 0:
                upper[cones.zero:cones.zero+cones.nonneg] = b[cones.zero:cones.zero+cones.nonneg]
            self.l = lower
            self.u = upper
