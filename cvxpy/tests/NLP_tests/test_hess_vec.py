import numpy as np

import cvxpy as cp


class TestHessVec():

    def test_log_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        log = cp.log(x)
        result_dict = log.hess_vec(vec)
        result_matrix = list(result_dict.values())[0]
        correct_matrix = np.diag([ -5.0/(1.0**2), -4.0/(2.0**2), -3.0/(3.0**2)])
        assert(np.allclose(result_matrix, correct_matrix))
    
    def test_sum_one_variable_log(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0])
        sum_log = cp.sum(cp.log(x))
        result_dict = sum_log.hess_vec(vec)
        result_matrix = list(result_dict.values())[0]
        result_correct = np.diag([ -5.0/(1.0**2), -5.0/(2.0**2), -5.0/(3.0**2)])
        assert(np.allclose(result_matrix, result_correct))

    def test_exp_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        exp = cp.exp(x)
        result_dict = exp.hess_vec(vec)
        result_matrix = list(result_dict.values())[0]
        correct_matrix = np.diag([ 5.0*np.exp(1.0), 4.0*np.exp(2.0), 3.0*np.exp(3.0)])
        assert(np.allclose(result_matrix, correct_matrix))

    def test_add_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([-1.0, 2, 4])
        sum_log = cp.log(x) + cp.log(x) + cp.exp(x)
        result_dict = sum_log.hess_vec(vec)
        result_matrix = list(result_dict.values())[0]
        result_correct = np.diag([-1 * (-2.0/(1.0**2) + np.exp(1.0)), 
                                2 * (-2.0/(2.0**2) + np.exp(2.0)), 
                                4 * (-2.0/(3.0**2) + np.exp(3.0))])
        assert(np.allclose(result_matrix, result_correct))

    def test_negation_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        neg_log = -cp.log(x)
        result_dict = neg_log.hess_vec(vec)
        result_matrix = list(result_dict.values())[0]
        correct_matrix = -np.diag([ -5.0/(1.0**2), -4.0/(2.0**2), -3.0/(3.0**2)])
        assert(np.allclose(result_matrix, correct_matrix))

    def test_multiply_with_constant_one_variable_test_one(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        scaled_log = 2.3 * cp.log(x)
        result_dict = scaled_log.hess_vec(vec)
        result_matrix = list(result_dict.values())[0]
        correct_matrix = 2.3 * np.diag([ -5.0/(1.0**2), -4.0/(2.0**2), -3.0/(3.0**2)])
        assert(np.allclose(result_matrix, correct_matrix))

    def test_multiply_with_constant_one_variable_test_two(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        scaled_log = cp.log(x) * 2.3
        result_dict = scaled_log.hess_vec(vec) 
        result_matrix = list(result_dict.values())[0]
        correct_matrix = 2.3 * np.diag([ -5.0/(1.0**2), -4.0/(2.0**2), -3.0/(3.0**2)])
        assert(np.allclose(result_matrix, correct_matrix))

    def test_add_one_variable_two_logs(self):
        pass 

    def test_add_two_variables_two_logs(self):
        pass




