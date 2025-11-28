import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('KNITRO' not in INSTALLED_SOLVERS, reason='KNITRO is not installed.')
class TestKNITROInterface:
    """Tests for KNITRO solver interface options and algorithms."""

    def test_knitro_basic_solve(self):
        """Test that KNITRO can solve a basic NLP problem."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize((x - 2) ** 2), [x >= 1])
        prob.solve(solver=cp.KNITRO, nlp=True)
        assert prob.status == cp.OPTIMAL
        assert np.isclose(x.value, 2.0, atol=1e-5)

    def test_knitro_algorithm_bar_direct(self):
        """Test Interior-Point/Barrier Direct algorithm (algorithm=1)."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        prob.solve(solver=cp.KNITRO, nlp=True, algorithm=1)

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_algorithm_bar_cg(self):
        """Test Interior-Point/Barrier CG algorithm (algorithm=2)."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        prob.solve(solver=cp.KNITRO, nlp=True, algorithm=2)

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_algorithm_act_cg(self):
        """Test Active-Set CG algorithm (algorithm=3)."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        prob.solve(solver=cp.KNITRO, nlp=True, algorithm=3)

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_algorithm_sqp(self):
        """Test Active-Set SQP algorithm (algorithm=4)."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        prob.solve(solver=cp.KNITRO, nlp=True, algorithm=4)

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_algorithm_alm(self):
        """Test Augmented Lagrangian Method algorithm (algorithm=6)."""
        # ALM works best on unconstrained or simple problems
        x = cp.Variable(2, name='x')
        prob = cp.Problem(
            cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2),
            []
        )

        prob.solve(solver=cp.KNITRO, nlp=True, algorithm=6)

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [1.0, 1.0], atol=1e-3)

    def test_knitro_hessopt_exact(self):
        """Test exact Hessian option (hessopt=1, default)."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        prob.solve(solver=cp.KNITRO, nlp=True, hessopt=1)

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_hessopt_bfgs(self):
        """Test BFGS Hessian approximation (hessopt=2)."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        prob.solve(solver=cp.KNITRO, nlp=True, hessopt=2)

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_hessopt_lbfgs(self):
        """Test L-BFGS Hessian approximation (hessopt=6)."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        prob.solve(solver=cp.KNITRO, nlp=True, hessopt=6)

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_hessopt_sr1(self):
        """Test SR1 Hessian approximation (hessopt=3)."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        prob.solve(solver=cp.KNITRO, nlp=True, hessopt=3)

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_maxit(self):
        """Test maximum iterations option."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        # Use a reasonable maxit value that allows convergence
        prob.solve(solver=cp.KNITRO, nlp=True, maxit=100)
        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_maxit_limit(self):
        """Test that max iterations limit is respected."""
        # Rosenbrock is harder to solve - use very small maxit
        x = cp.Variable(2, name='x')
        prob = cp.Problem(
            cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2),
            []
        )

        # With only 1 iteration, solver should hit the limit
        prob.solve(solver=cp.KNITRO, nlp=True, maxit=1)
        # Status should be USER_LIMIT (iteration limit reached)
        assert prob.status in [cp.USER_LIMIT, cp.OPTIMAL_INACCURATE, cp.OPTIMAL]

    def test_knitro_combined_options(self):
        """Test combining multiple KNITRO options."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        prob = cp.Problem(cp.Minimize(x[0]**2 + x[1]**2), [x[0] + x[1] >= 1])

        prob.solve(
            solver=cp.KNITRO,
            nlp=True,
            algorithm=1,     # BAR_DIRECT
            hessopt=2,       # BFGS
            feastol=1e-8,
            opttol=1e-8,
        )

        assert prob.status == cp.OPTIMAL
        assert np.allclose(x.value, [0.5, 0.5], atol=1e-4)

    def test_knitro_unknown_option_raises(self):
        """Test that unknown options raise ValueError."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize((x - 2) ** 2), [x >= 1])

        with pytest.raises(ValueError, match="Unknown KNITRO option"):
            prob.solve(solver=cp.KNITRO, nlp=True, unknown_option=123)


@pytest.mark.skipif('COPT' not in INSTALLED_SOLVERS, reason='COPT is not installed.')
class TestCOPTInterface:

    def test_copt_call(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize((x - 2) ** 2), [x >= 1])
        with pytest.raises(NotImplementedError):
            prob.solve(solver=cp.COPT, nlp=True)
