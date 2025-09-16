import numpy as np
import pytest

import cvxpy as cp


class TestHessVecMultiply():

    # x * y with x constant, y variable
    def test_hess_vec_multiply_one_constant_test_one(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        mult = cp.multiply(5.0, x)
        vec = np.array([5.0, 4.0, 3.0])
        result_dict = mult.hess_vec(vec)
        assert(result_dict == {})

    # x * y with x variable, y constant
    def test_hess_vec_multiply_one_constant_test_two(self):
        n = 3 
        y = cp.Variable((n, ), name='y')
        y.value = np.array([1.0, 2.0, 3.0])
        mult = cp.multiply(y, 5.0)
        vec = np.array([5.0, 4.0, 3.0])
        result_dict = mult.hess_vec(vec)
        assert(result_dict == {})

    # x * y with x variable, should raise an error
    def test_hess_vec_multiply_same_variable(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        mult = cp.multiply(x, x)
        vec = np.array([5.0, 4.0, 3.0])
        with pytest.raises(AssertionError):
            mult.hess_vec(vec)

    # x * y with x vector variable, y vector variable
    def test_hess_vec_multiply_two_vector_variables_same_dimension(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        y = cp.Variable((3, ), name='y')
        y.value = np.array([4.0, 5.0, 6.0])
        mult = cp.multiply(x, y)
        vec = np.array([5.0, 4.0, 3.0])
        result_dict = mult.hess_vec(vec)
        correct_matrix = np.diag(np.array([5.0, 4.0, 3.0]))
        assert(np.allclose(result_dict[(x, y)], correct_matrix))
        assert(np.allclose(result_dict[(y, x)], correct_matrix))
        assert(len(result_dict) == 2)

    # x * y with x scalar variable, y scalar variable
    def test_hess_vec_multiply_two_scalar_variables_same_dimension(self):
        x = cp.Variable((1, ), name='x')
        x.value = np.array([3.0])
        y = cp.Variable((1, ), name='y')
        y.value = np.array([4.0])
        mult = cp.multiply(x, y)
        vec = np.array([5.0])
        result_dict = mult.hess_vec(vec)
        correct_matrix = np.array([[5.0]])
        assert(np.allclose(result_dict[(x, y)], correct_matrix))
        assert(np.allclose(result_dict[(y, x)], correct_matrix))
        assert(len(result_dict) == 2)

    # x * y with x vector variable, y scalar variable,
    def test_hess_vec_multiply_two_variables_broadcasting_test_one(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        y = cp.Variable((1, ), name='y')
        y.value = np.array([4.0])
        mult = cp.multiply(x, y)
        vec = np.array([5.0, 4.0, 3.0])
        result_dict = mult.hess_vec(vec)
        correct_matrix = np.array([5.0, 4.0, 3.0])
        assert(np.allclose(result_dict[(x, y)], correct_matrix))
        assert(np.allclose(result_dict[(y, x)], correct_matrix))
        assert(len(result_dict) == 2)

    # x * y with x scalar variable, y vector variable,
    def test_hess_vec_multiply_two_variables_broadcasting_test_two(self):
        x = cp.Variable((1, ), name='x')
        x.value = np.array([4.0])
        y = cp.Variable((3, ), name='y')
        y.value = np.array([1.0, 2.0, 3.0])
        mult = cp.multiply(x, y)
        vec = np.array([5.0, 4.0, 3.0])
        result_dict = mult.hess_vec(vec)
        correct_matrix = np.array([5.0, 4.0, 3.0])
        assert(np.allclose(result_dict[(x, y)], correct_matrix))
        assert(np.allclose(result_dict[(y, x)], correct_matrix))
        assert(len(result_dict) == 2)




