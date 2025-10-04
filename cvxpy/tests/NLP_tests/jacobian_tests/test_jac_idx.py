import numpy as np

import cvxpy as cp


class TestJacobianIndex():


    def test_jacobian_simple_idx(self):
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x)[1]
        result_dict = expr.jacobian()
        correct_jacobian = np.zeros((1, n))
        correct_jacobian[0, 1] = 1/x.value[1]
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_jacobian_array_idx(self):
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        idxs = np.array([0, 2])

    def test_jacobian_slice_idx(self):
        pass

    def test_jacobian_special_idx(self):
        pass

    def test_jacobian_matrix_slice(self):
        pass
