import cvxpy as cp

x = cp.Variable(3)

objective = cp.Minimize(-x[0]*x[1] - x[1]*x[2])
constraints = [
    x[0]**2 - x[1]**2 + x[2]**2 - 2 <= 0,
    x[0]**2 + x[1]**2 + x[2]**2 - 10 <= 0
]

prob = cp.Problem(objective, constraints)
prob.solve(nlp=True, solver=cp.PIPS)  # any NLP-capable solver; default usually works
print("Status:", prob.status)
print("Objective:", prob.value)
print("x:", x.value)
