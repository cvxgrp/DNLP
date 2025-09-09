import numpy as np 
import numpy.linalg as LA
import cvxpy as cp
from cvxpy.reductions.expr2smooth.expr2smooth import Expr2Smooth

np.random.seed(1234)
TOL = 1e-3
mu = cp.Variable((1, ), name="mu")

for n in [42]:
    np.random.seed(1234)
    data = np.random.randn(n)
    sigma_opt = (1 / np.sqrt(n)) * LA.norm(data - np.mean(data))
    mu_opt = np.mean(data)
    mu.value = None
    print("n, scale factor: ", n)
    sigma = cp.Variable((1, ), nonneg=True, name="sigma")
    obj = (n / 2) * cp.log(2*np.pi*cp.square(sigma)) + (1 / (2 * cp.square(sigma))) * cp.sum(cp.square(data-mu))
    constraints = []

    problem = cp.Problem(cp.Minimize(obj), constraints)
    reduction = Expr2Smooth(problem)
    new_prob, inv_data = reduction.apply(problem)
    print(str(new_prob))

    problem.solve(solver=cp.IPOPT, nlp=True)

    print("sigma.value: ", sigma.value)
    print("sigma_opt: ", sigma_opt)
    assert(np.abs(sigma.value - sigma_opt) / np.max([1, np.abs(sigma_opt)]) <= TOL)
    assert(np.abs(mu.value - mu_opt) / np.max([1, np.abs(mu_opt)]) <= TOL)


#print(str(new_prob))

print("sigma.value: ", sigma.value)
print("sigma_opt: ", sigma_opt)
assert(np.abs(sigma.value - sigma_opt) / np.max([1, np.abs(sigma_opt)]) <= TOL)
assert(np.abs(mu.value - mu_opt) / np.max([1, np.abs(mu_opt)]) <= TOL)