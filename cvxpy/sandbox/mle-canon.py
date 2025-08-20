import numpy as np

import cvxpy as cp

n = 2
np.random.seed(1234)
data = np.random.randn(n)

mu = cp.Variable(1, name="mu")
mu.value = np.array([0.0])
sigma = cp.Variable(1, name="sigma")
sigma.value = np.array([1.0])
t1 = cp.Variable(1, name="t1")
t2 = cp.Variable(n, name="t2")
t3 = cp.Variable(1, name="t3")
t4 = cp.Variable(1, name="t4")
t5 = cp.Variable(1, name="t5")

log_likelihood = ((n / 2) * cp.log(t4) - cp.sum(cp.square(t2)) / (2 * (t1)**2))

constraints = [
    t1 == sigma,
    t2 == data - mu,
    t3 == (2 * np.pi * (t5)**2),
    t4 == 1 / t3,
    t5 == sigma,
]
t1.value = sigma.value
t2.value = data - mu.value
t3.value = (2 * np.pi * (sigma.value)**2)
t4.value = 1 / t3.value
t5.value = sigma.value

objective = cp.Maximize(log_likelihood)
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.IPOPT, nlp=True)
assert problem.status == cp.OPTIMAL
assert np.allclose(mu.value, np.mean(data))
assert np.allclose(sigma.value, np.std(data))
