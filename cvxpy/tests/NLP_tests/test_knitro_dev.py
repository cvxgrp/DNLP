"""
Development test file for KNITRO NLP interface.
"""
import numpy as np
import cvxpy as cp


def test_simple_qp():
    """Simple quadratic problem."""
    print("\n" + "="*60)
    print("TEST: Simple QP")
    print("="*60)
    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])

    objective = cp.Minimize(x[0]**2 + x[1]**2)
    constraints = [x[0] + x[1] >= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-5)
    print("PASSED!")


def test_hs071():
    """Classic HS071 problem from IPOPT documentation."""
    print("\n" + "="*60)
    print("TEST: HS071")
    print("="*60)
    x = cp.Variable(4, bounds=[0, 6])
    x.value = np.array([1.0, 5.0, 5.0, 1.0])
    objective = cp.Minimize(x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2])

    constraints = [
        x[0]*x[1]*x[2]*x[3] >= 25,
        cp.sum(cp.square(x)) == 40,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.75450865, 4.63936861, 3.78856881, 1.88513184]), atol=1e-3)
    print("PASSED!")


def test_rosenbrock():
    """Rosenbrock function - unconstrained."""
    print("\n" + "="*60)
    print("TEST: Rosenbrock")
    print("="*60)
    x = cp.Variable(2, name='x')
    objective = cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)
    problem = cp.Problem(objective, [])
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 1.0]), atol=1e-4)
    print("PASSED!")


def test_portfolio_opt():
    """Portfolio optimization with quadratic form."""
    print("\n" + "="*60)
    print("TEST: Portfolio Optimization")
    print("="*60)
    r = np.array([0.026002150277777, 0.008101316405671, 0.073715909491990])
    Q = np.array([
        [0.018641039983891, 0.003598532927677, 0.001309759253660],
        [0.003598532927677, 0.006436938322676, 0.004887265158407],
        [0.001309759253660, 0.004887265158407, 0.068682765454814],
    ])
    x = cp.Variable(3)
    x.value = np.array([10.0, 10.0, 10.0])
    variance = cp.quad_form(x, Q)
    expected_return = r @ x
    problem = cp.Problem(
        cp.Minimize(variance),
        [
            cp.sum(x) <= 1000,
            expected_return >= 50,
            x >= 0
        ]
    )
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    print("PASSED!")


def test_qcp():
    """Quadratically constrained problem."""
    print("\n" + "="*60)
    print("TEST: QCP")
    print("="*60)
    x = cp.Variable(1)
    y = cp.Variable(1, bounds=[0, np.inf])
    z = cp.Variable(1, bounds=[0, np.inf])

    objective = cp.Maximize(x)

    constraints = [
        x + y + z == 1,
        x**2 + y**2 - z**2 <= 0,
        x**2 - cp.multiply(y, z) <= 0
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}, y = {y.value}, z = {z.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.32699284]), atol=1e-4)
    print("PASSED!")


def test_geo_mean():
    """Geometric mean maximization."""
    print("\n" + "="*60)
    print("TEST: Geometric Mean")
    print("="*60)
    x = cp.Variable(3, pos=True)
    geo_mean = cp.geo_mean(x)
    objective = cp.Maximize(geo_mean)
    constraints = [cp.sum(x) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([1/3, 1/3, 1/3]), atol=1e-4)
    print("PASSED!")


def test_socp():
    """Second-order cone program."""
    print("\n" + "="*60)
    print("TEST: SOCP")
    print("="*60)
    x = cp.Variable(3)
    y = cp.Variable()

    objective = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])

    constraints = [
        cp.norm(x, 2) <= y,
        x[0] + x[1] + 3*x[2] >= 1.0,
        y <= 5
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}, y = {y.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(objective.value, -13.548638814247532, atol=1e-3)
    print("PASSED!")


def test_mle():
    """Maximum likelihood estimation."""
    print("\n" + "="*60)
    print("TEST: MLE")
    print("="*60)
    n = 1000
    np.random.seed(1234)
    data = np.random.randn(n)

    mu = cp.Variable((1,), name="mu")
    mu.value = np.array([0.0])
    sigma = cp.Variable((1,), name="sigma")
    sigma.value = np.array([1.0])

    constraints = [mu == sigma**2]
    log_likelihood = (
        (n / 2) * cp.log(1 / (2 * np.pi * (sigma)**2))
        - cp.sum(cp.square(data-mu)) / (2 * (sigma)**2)
    )

    objective = cp.Maximize(log_likelihood)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"sigma = {sigma.value}, mu = {mu.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(sigma.value, 0.77079388, atol=1e-3)
    assert np.allclose(mu.value, 0.59412321, atol=1e-3)
    print("PASSED!")


def test_analytic_polytope_center():
    """Analytic center of a polytope."""
    print("\n" + "="*60)
    print("TEST: Analytic Polytope Center")
    print("="*60)
    np.random.seed(0)
    m, n = 50, 4
    b = np.ones(m)
    rand = np.random.randn(m - 2*n, n)
    A = np.vstack((rand, np.eye(n), np.eye(n) * -1))

    x = cp.Variable(n)
    objective = cp.Minimize(-cp.sum(cp.log(b - A @ x)))
    problem = cp.Problem(objective, [])
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    print("PASSED!")


def test_localization():
    """Localization problem with sqrt constraints."""
    print("\n" + "="*60)
    print("TEST: Localization")
    print("="*60)
    np.random.seed(42)
    m = 10
    dim = 2
    x_true = np.array([2.0, -1.5])
    a = np.random.uniform(-5, 5, (m, dim))
    rho = np.linalg.norm(a - x_true, axis=1)

    x = cp.Variable(2, name='x')
    t = cp.Variable(m, name='t')
    constraints = [t == cp.sqrt(cp.sum(cp.square(x - a), axis=1))]
    objective = cp.Minimize(cp.sum_squares(t - rho))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"x = {x.value}, x_true = {x_true}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, x_true, atol=1e-3)
    print("PASSED!")


def test_algorithm_bar_direct():
    """Test Interior-Point/Barrier Direct algorithm."""
    print("\n" + "="*60)
    print("TEST: Algorithm - Interior-Point/Barrier Direct")
    print("="*60)

    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])
    objective = cp.Minimize(x[0]**2 + x[1]**2)
    constraints = [x[0] + x[1] >= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        algorithm=1  # KN_ALG_BAR_DIRECT
    )

    print(f"Status: {problem.status}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-4)
    print("PASSED!")


def test_algorithm_bar_cg():
    """Test Interior-Point/Barrier CG algorithm."""
    print("\n" + "="*60)
    print("TEST: Algorithm - Interior-Point/Barrier CG")
    print("="*60)

    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])
    objective = cp.Minimize(x[0]**2 + x[1]**2)
    constraints = [x[0] + x[1] >= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        algorithm=2  # KN_ALG_BAR_CG
    )

    print(f"Status: {problem.status}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-4)
    print("PASSED!")


def test_algorithm_alm():
    """Test Augmented Lagrangian Method (ALM) algorithm."""
    print("\n" + "="*60)
    print("TEST: Algorithm - Augmented Lagrangian Method (ALM)")
    print("="*60)

    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])
    objective = cp.Minimize(x[0]**2 + x[1]**2)
    constraints = [x[0] + x[1] >= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        algorithm=6  # KN_ALG_AL (Augmented Lagrangian)
    )

    print(f"Status: {problem.status}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-4)
    print("PASSED!")


def test_algorithm_sqp():
    """Test Active-Set SQP algorithm."""
    print("\n" + "="*60)
    print("TEST: Algorithm - Active-Set SQP")
    print("="*60)

    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])
    objective = cp.Minimize(x[0]**2 + x[1]**2)
    constraints = [x[0] + x[1] >= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        algorithm=4  # KN_ALG_ACT_SQP
    )

    print(f"Status: {problem.status}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-4)
    print("PASSED!")


def test_tolerances():
    """Test custom tolerance settings."""
    print("\n" + "="*60)
    print("TEST: Custom Tolerances")
    print("="*60)

    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])
    objective = cp.Minimize(x[0]**2 + x[1]**2)
    constraints = [x[0] + x[1] >= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        feastol=1e-10,
        opttol=1e-10,
    )

    print(f"Status: {problem.status}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-8)
    print("PASSED!")


def test_alm_on_rosenbrock():
    """Test ALM algorithm on Rosenbrock problem (unconstrained works better)."""
    print("\n" + "="*60)
    print("TEST: ALM on Rosenbrock")
    print("="*60)

    x = cp.Variable(2, name='x')
    objective = cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)
    problem = cp.Problem(objective, [])
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        algorithm=6  # KN_ALG_AL
    )

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 1.0]), atol=1e-3)
    print("PASSED!")


def test_max_iterations():
    """Test max iterations setting."""
    print("\n" + "="*60)
    print("TEST: Max Iterations")
    print("="*60)

    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])
    objective = cp.Minimize(x[0]**2 + x[1]**2)
    constraints = [x[0] + x[1] >= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        maxit=100
    )

    print(f"Status: {problem.status}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    print("PASSED!")


def test_alm_simple_qp():
    """Test ALM on simple QP."""
    print("\n" + "="*60)
    print("TEST: ALM on Simple QP")
    print("="*60)

    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])
    objective = cp.Minimize(x[0]**2 + x[1]**2)
    constraints = [x[0] + x[1] >= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        algorithm=6  # KN_ALG_AL
    )

    print(f"Status: {problem.status}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-4)
    print("PASSED!")


def test_alm_portfolio():
    """Test ALM on portfolio optimization."""
    print("\n" + "="*60)
    print("TEST: ALM on Portfolio Optimization")
    print("="*60)

    r = np.array([0.026002150277777, 0.008101316405671, 0.073715909491990])
    Q = np.array([
        [0.018641039983891, 0.003598532927677, 0.001309759253660],
        [0.003598532927677, 0.006436938322676, 0.004887265158407],
        [0.001309759253660, 0.004887265158407, 0.068682765454814],
    ])
    x = cp.Variable(3)
    x.value = np.array([10.0, 10.0, 10.0])
    variance = cp.quad_form(x, Q)
    expected_return = r @ x
    problem = cp.Problem(
        cp.Minimize(variance),
        [
            cp.sum(x) <= 1000,
            expected_return >= 50,
            x >= 0
        ]
    )
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        algorithm=6  # KN_ALG_AL
    )

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    print("PASSED!")


def test_alm_qcp():
    """Test ALM on quadratically constrained problem."""
    print("\n" + "="*60)
    print("TEST: ALM on QCP")
    print("="*60)

    x = cp.Variable(1)
    y = cp.Variable(1, bounds=[0, np.inf])
    z = cp.Variable(1, bounds=[0, np.inf])

    objective = cp.Maximize(x)

    constraints = [
        x + y + z == 1,
        x**2 + y**2 - z**2 <= 0,
        x**2 - cp.multiply(y, z) <= 0
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        algorithm=6  # KN_ALG_AL
    )

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}, y = {y.value}, z = {z.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.32699284]), atol=1e-3)
    print("PASSED!")


def test_alm_geo_mean():
    """Test ALM on geometric mean maximization."""
    print("\n" + "="*60)
    print("TEST: ALM on Geometric Mean")
    print("="*60)

    x = cp.Variable(3, pos=True)
    geo_mean = cp.geo_mean(x)
    objective = cp.Maximize(geo_mean)
    constraints = [cp.sum(x) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        algorithm=6  # KN_ALG_AL
    )

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([1/3, 1/3, 1/3]), atol=1e-3)
    print("PASSED!")


def test_exact_hessian():
    """Test that exact Hessian is being used by default."""
    print("\n" + "="*60)
    print("TEST: Exact Hessian (default)")
    print("="*60)

    # Rosenbrock is a good test - exact Hessian helps convergence
    x = cp.Variable(2, name='x')
    objective = cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)
    problem = cp.Problem(objective, [])
    problem.solve(solver=cp.KNITRO, nlp=True, verbose=True)

    print(f"Status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 1.0]), atol=1e-4)
    print("PASSED!")


def test_bfgs_hessian():
    """Test BFGS Hessian approximation option."""
    print("\n" + "="*60)
    print("TEST: BFGS Hessian Approximation")
    print("="*60)

    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])
    objective = cp.Minimize(x[0]**2 + x[1]**2)
    constraints = [x[0] + x[1] >= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.KNITRO,
        nlp=True,
        verbose=True,
        hessopt=2  # KN_HESSOPT_BFGS
    )

    print(f"Status: {problem.status}")
    print(f"x = {x.value}")
    assert problem.status == cp.OPTIMAL
    assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-4)
    print("PASSED!")


if __name__ == "__main__":
    # Basic tests
    test_simple_qp()
    test_hs071()
    test_rosenbrock()
    test_portfolio_opt()
    test_qcp()
    test_geo_mean()
    test_socp()
    test_mle()
    test_analytic_polytope_center()
    test_localization()

    # Algorithm tests
    test_algorithm_bar_direct()
    test_algorithm_bar_cg()
    test_algorithm_alm()
    test_algorithm_sqp()

    # Options tests
    test_tolerances()
    test_alm_on_rosenbrock()
    test_max_iterations()

    # Additional ALM tests
    test_alm_simple_qp()
    test_alm_portfolio()
    test_alm_qcp()
    test_alm_geo_mean()

    # Hessian tests
    test_exact_hessian()
    test_bfgs_hessian()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
