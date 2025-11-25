import numpy as np

import cvxpy as cp


class TestHessTranspose():


    def test_vec(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x).T
        result_dict = expr.hess_vec(np.ones(n))
        correct_hess = -np.diag(1/x.value)**2
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_hess))
       
    def test_col_vec(self):
        n = 3 
        x = cp.Variable((n, 1), name='x')
        x.value = np.array([1.0, 2.0, 3.0]).reshape((n, 1), order='F')
        expr = cp.log(x).T
        result_dict = expr.hess_vec(np.ones(n))
        correct_hess = -np.diag(1/x.value.reshape((n)))**2

        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_hess))

    def test_mat(self):
        n = 3
        m = 2
        x = cp.Variable((3, 2), name='x')
        x.value = np.array([[1.0, 2.0,], [3.0, 4.0], [5.0, 6.0]])
        expr = cp.log(x).T
        result_dict = expr.hess_vec(np.ones(n * m))
        correct_hess = -np.array([[1., 0., 0., 0., 0., 0.,],
                                     [0., 0., 0.2, 0., 0., 0.,],
                                     [0., 0., 0., 0., 0.25, 0.,],
                                     [0., 0.33333333, 0., 0., 0., 0.,],
                                     [0., 0., 0., 0.5, 0., 0.,],
                                     [0., 0., 0., 0., 0., 0.16666667]]
        )**2

        
        computed_hess = np.zeros((n * m, n * m))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_hess))
