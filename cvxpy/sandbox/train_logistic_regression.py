import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cvxpy as cp


def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + cp.exp(-x))

def relu(x):
    """ReLU activation function"""
    return cp.maximum(0, x)

def train_two_layer_nn(X, y, hidden_dim=10, lambda_reg=0.01, activation='sigmoid'):
    """
    Train a two-layer neural network using CVXPY with logistic regression loss.
    Assumes CVXPY can handle non-convex problems via NLP solvers.
    
    Parameters:
    -----------
    X : numpy array of shape (n_samples, n_features)
        Training data
    y : numpy array of shape (n_samples,)
        Binary labels (0 or 1)
    hidden_dim : int
        Number of hidden units
    lambda_reg : float
        L2 regularization parameter
    activation : str
        Activation function ('relu' or 'sigmoid')
    
    Returns:
    --------
    W1, b1, W2, b2 : Trained network parameters
    problem : CVXPY problem object
    """
    n_samples, n_features = X.shape
    
    # Define optimization variables for network parameters
    # W1: list of n_features-dimensional vectors, one per hidden unit
    W1_vecs = [cp.Variable(n_features) for _ in range(hidden_dim)]  # First layer weights (list of vectors)
    W1 = cp.hstack(W1_vecs)  # Shape: (n_features, hidden_dim)
    b1 = cp.Variable(hidden_dim)                 # First layer bias (1-d)
    W2 = cp.Variable(hidden_dim)                 # Second layer weights (1-d)
    b2 = cp.Variable(1)                          # Second layer bias (1-d)
    
    # Forward pass through the network
    # First layer: linear transformation
    # Z1: shape (n_samples, hidden_dim)
    Z1 = X @ W1 + b1  # W1 is now (n_features, hidden_dim)
    
    # Apply activation function
    if activation == 'relu':
        H1 = relu(Z1)
    elif activation == 'sigmoid':
        H1 = sigmoid(Z1)
    else:
        H1 = Z1  # Linear activation
    
    # Second layer: linear transformation
    Z2 = H1 @ W2 + b2  # H1 shape (n_samples, hidden_dim), W2 shape (hidden_dim,)
    
    # Output probabilities using sigmoid
    y_pred = sigmoid(Z2)
    
    # Logistic regression loss (binary cross-entropy)
    # Loss = -1/n * sum(y*log(y_pred) + (1-y)*log(1-y_pred))
    y_reshaped = y.reshape(-1, 1)
    logistic_loss = -cp.sum(
        cp.multiply(y_reshaped, cp.log(y_pred + 1e-8)) + 
        cp.multiply(1 - y_reshaped, cp.log(1 - y_pred + 1e-8))
    ) / n_samples
    
    # L2 regularization
    l2_penalty = lambda_reg * (
        sum(cp.sum_squares(w) for w in W1_vecs) + 
        cp.sum_squares(W2) + 
        cp.sum_squares(b1) + 
        cp.sum_squares(b2)
    )
    
    # Total objective function
    objective = cp.Minimize(logistic_loss + l2_penalty)
    
    # Define the optimization problem
    problem = cp.Problem(objective)
    
    # Solve using a nonlinear programming solver
    # Assuming CVXPY can interface with NLP solvers like IPOPT, SNOPT, etc.
    problem.solve(solver='IPOPT', nlp=True, verbose=True)
    
    return W1.value, b1.value, W2.value, b2.value, problem

def predict(X, W1, b1, W2, b2, activation='sigmoid'):
    """
    Make predictions using the trained neural network.
    
    Parameters:
    -----------
    X : numpy array
        Input data
    W1, b1, W2, b2 : numpy arrays
        Network parameters
    activation : str
        Activation function
    
    Returns:
    --------
    predictions : numpy array
        Binary predictions
    probabilities : numpy array
        Predicted probabilities
    """
    # First layer
    # W1: list of vectors
    Z1 = np.column_stack([X @ W1[j] + b1[j] for j in range(len(W1))])
    
    # Activation
    if activation == 'relu':
        H1 = np.maximum(0, Z1)
    elif activation == 'sigmoid':
        H1 = 1 / (1 + np.exp(-Z1))
    else:
        H1 = Z1
    
    # Second layer
    Z2 = H1 @ W2 + b2  # H1 shape (n_samples, hidden_dim), W2 shape (hidden_dim,)
    
    # Output probabilities
    probabilities = 1 / (1 + np.exp(-Z2))
    
    # Binary predictions
    predictions = (probabilities > 0.5).astype(int)
    
    return predictions.flatten(), probabilities.flatten()

# Example usage
if __name__ == "__main__":
    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the neural network
    print("Training two-layer neural network with CVXPY...")
    print("=" * 50)
    
    W1, b1, W2, b2, problem = train_two_layer_nn(
        X_train, y_train, 
        hidden_dim=20, 
        lambda_reg=0.001,
        activation='sigmoid'
    )
    
    print(f"\nOptimization status: {problem.status}")
    print(f"Final loss value: {problem.value:.4f}")
    
    # Make predictions on test set
    y_pred, y_prob = predict(X_test, W1, b1, W2, b2, activation='sigmoid')

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Print network architecture summary
    print("\nNetwork Architecture:")
    print(f"Input layer: {X_train.shape[1]} features")
    print(f"Hidden layer: {W1.shape[1]} units (ReLU activation)")
    print("Output layer: 1 unit (Sigmoid activation)")
    print(f"\nTotal parameters: {W1.size + b1.size + W2.size + b2.size}")
    
    # Additional training with different activation
    print("\n" + "=" * 50)
    print("Training with sigmoid activation...")
    
    W1_sig, b1_sig, W2_sig, b2_sig, problem_sig = train_two_layer_nn(
        X_train, y_train,
        hidden_dim=20,
        lambda_reg=0.001,
        activation='sigmoid'
    )
    
    y_pred_sig, y_prob_sig = predict(X_test, W1_sig, b1_sig, W2_sig, b2_sig, activation='sigmoid')
    accuracy_sig = np.mean(y_pred_sig == y_test)
    
    print(f"Test accuracy (sigmoid): {accuracy_sig:.4f}")
    
    # Compare the two models
    print("\nModel Comparison:")
    print(f"ReLU activation accuracy: {accuracy:.4f}")
    print(f"Sigmoid activation accuracy: {accuracy_sig:.4f}")
