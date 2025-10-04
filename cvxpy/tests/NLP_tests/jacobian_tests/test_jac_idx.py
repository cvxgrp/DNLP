import numpy as np

import cvxpy as cp


class TestJacobianIndex():


    def test_two_atoms_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        sum = cp.log(x) + cp.exp(x) + 7 * cp.log(x)
        result_dict = sum.jacobian()
        correct_jacobian = np.diag(1/x.value) + np.diag(np.exp(x.value)) + np.diag(7/x.value)
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
