import cvxpy as cp

# minimize square root of x
x = cp.Variable(bounds=[1e-8, None])
obj = cp.Minimize(cp.sqrt(x))
constraints = []
problem = cp.Problem(obj, constraints)
problem.solve(solver=cp.KNITRO, nlp=True, honorbnds=1, verbose=True, outlev=3, algorithm=6)