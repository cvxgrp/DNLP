import cvxpy as cp
import numpy as np

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 50

# Create two classes
X_class0 = np.random.randn(n_samples // 2, n_features) - 1
X_class1 = np.random.randn(n_samples // 2, n_features) + 1
X = np.vstack([X_class0, X_class1])

# Labels: -1 for class 0, +1 for class 1
y = np.concatenate([-np.ones(n_samples // 2), np.ones(n_samples // 2)])

# Add intercept term (bias)
X_with_intercept = np.column_stack([np.ones(n_samples), X])

# Define CVXPY variables
n_features_with_intercept = n_features + 1
w = cp.Variable(n_features_with_intercept)  # weights including bias

# Regularization parameter
lambda_reg = 0.1

# Logistic regression objective: minimize negative log-likelihood + L2 regularization
# Using log-sum-exp formulation for numerical stability
log_likelihood = cp.sum(cp.logistic(-cp.multiply(y, X_with_intercept @ w)))
regularization = lambda_reg * cp.norm(w[1:], 2)**2  # Don't regularize intercept

# Define the optimization problem
objective = cp.Minimize(log_likelihood + regularization)
problem = cp.Problem(objective)

# Solve the problem
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

# Print results
print(f"Optimization status: {problem.status}")
print(f"Optimal objective value: {problem.value:.4f}")
print("Optimal weights (including bias):")
print(f"  Bias (intercept): {w.value[0]:.4f}")
for i in range(n_features):
    print(f"  Weight {i+1}: {w.value[i+1]:.4f}")

# Make predictions on training data
predictions = np.sign(X_with_intercept @ w.value)
accuracy = np.mean(predictions == y)
print(f"\nTraining accuracy: {accuracy:.2%}")
