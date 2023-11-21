import numpy as np
import pandas as pd
from scipy.optimize import minimize
from cvxopt import matrix, solvers

# Function to calculate sub-gradient
def sub_gradient(w, b, x, y, C):
    hinge_loss = 1 - y * (np.dot(w, x) + b)
    if hinge_loss <= 0:
        grad_w = w
        grad_b = 0
    else:
        grad_w = w - C * y * x
        grad_b = -C * y

    return grad_w, grad_b

# Function to update weights
def update_weights(w, b, x, y, eta, grad_w, grad_b):
    w -= eta * grad_w
    b -= eta * grad_b
    return w, b

# Function to compute objective function value
def objective_function(w, b, X, y, C):
    hinge_losses = np.maximum(0, 1 - y * (np.dot(X, w) + b))
    regularization_term = 0.5 * np.dot(w, w)
    return np.mean(hinge_losses) + C * regularization_term

# Shuffling function
def shuffle(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def schedule2A(C_values, gamma_0_values, a_values, T, X_train, y_train, X_test, y_test):
    tr = []
    te = []
    gradW = []
    gradB = []

    for C in C_values:
            for gamma_0 in gamma_0_values:
                for a in a_values:
                    w = np.zeros(X_train.shape[1])  # Initialize weights
                    b = 0  # Initialize bias

                    eta = gamma_0  # Initial learning rate
                    updates = 0  # Counter for updates

                    for epoch in range(T):
                        # everyday I'm shuffling 
                        X_train, y_train = shuffle(X_train, y_train)  
                        for i, x in enumerate(X_train):
                            grad_w, grad_b = sub_gradient(w, b, x, y_train[i], C)
                            w, b = update_weights(w, b, x, y_train[i], eta, grad_w, grad_b)
                            updates += 1

                            # Update learning rate
                            eta = gamma_0 / (1 + (gamma_0 / a) * updates)

                        # Compute objective function value after each epoch
                        train_obj = objective_function(w, b, X_train, y_train, C)
                        test_obj = objective_function(w, b, X_test, y_test, C)
                        # print(f"Epoch: {epoch + 1}, Training Objective: {train_obj}, Test Objective: {test_obj}, Updates: {updates}")

                    # Calculate training and test error
                    train_error = np.mean(np.sign(np.dot(X_train, w) + b) != y_train)
                    test_error = np.mean(np.sign(np.dot(X_test, w) + b) != y_test)

                    # Saving parameters
                    tr.append(train_error)
                    te.append(test_error)
                    gradW.append(w)
                    gradB.append(b)

        # Report training and test error for this setting of C, gamma_0, and a
                # print(f"For C={C}, gamma_0={gamma_0}, a={a}:")
                # print(f"Training Error: {train_error}, Test Error: {test_error}")
    print(len(tr))
    return tr, te, gradW, gradB

def schedule2B(C_values, gamma_0_values, a_values, T, X_train, y_train, X_test, y_test):
    tr = []
    te = []
    gradW = []
    gradB = []

    for C in C_values:
            for gamma_0 in gamma_0_values:
                for a in a_values:
                    w = np.zeros(X_train.shape[1])  # Initialize weights
                    b = 0  # Initialize bias

                    eta = gamma_0  # Initial learning rate
                    updates = 0  # Counter for updates

                    for epoch in range(T):
                        # everyday I'm shuffling 
                        X_train, y_train = shuffle(X_train, y_train)  
                        for i, x in enumerate(X_train):
                            grad_w, grad_b = sub_gradient(w, b, x, y_train[i], C)
                            w, b = update_weights(w, b, x, y_train[i], eta, grad_w, grad_b)
                            updates += 1

                            # Update learning rate
                            eta = gamma_0 / (1 + updates)

                        # Compute objective function value after each epoch
                        train_obj = objective_function(w, b, X_train, y_train, C)
                        test_obj = objective_function(w, b, X_test, y_test, C)
                        # print(f"Epoch: {epoch + 1}, Training Objective: {train_obj}, Test Objective: {test_obj}, Updates: {updates}")

                    # Calculate training and test error
                    train_error = np.mean(np.sign(np.dot(X_train, w) + b) != y_train)
                    test_error = np.mean(np.sign(np.dot(X_test, w) + b) != y_test)

                    tr.append(train_error)
                    te.append(test_error)
                    gradW.append(w)
                    gradB.append(b)

        # Report training and test error for this setting of C, gamma_0, and a
                # print(f"For C={C}, gamma_0={gamma_0}, a={a}:")
                # print(f"Training Error: {train_error}, Test Error: {test_error}")
    print(len(tr))
    return tr, te, gradW, gradB

def train_svm(X_train, y_train, X_test, y_test, C, gamma_0, a, T=100):
    w = np.zeros(X_train.shape[1])  # Initialize weights
    b = 0  # Initialize bias
    eta = gamma_0  # Initial learning rate
    updates = 0  # Counter for updates

    train_obj_values = []  # Store training objective values
    test_obj_values = []  # Store test objective values

    for epoch in range(T):
        X_train, y_train = shuffle(X_train, y_train)  # Shuffle data
        for i, x in enumerate(X_train):
            grad_w, grad_b = sub_gradient(w, b, x, y_train[i], C)
            w, b = update_weights(w, b, x, y_train[i], eta, grad_w, grad_b)
            updates += 1

            # Update learning rate
            eta = gamma_0 / (1 + (gamma_0 / a) * updates)

        # Compute objective function value after each epoch
        train_obj = objective_function(w, b, X_train, y_train, C)
        test_obj = objective_function(w, b, X_test, y_test, C)
        train_obj_values.append(train_obj)
        test_obj_values.append(test_obj)

    return train_obj_values, test_obj_values

# Function to calculate the dual SVM objective
def dual_objective(alpha, X, y, C):
    n_samples = X.shape[0]
    alpha_sum = np.sum(alpha)
    alpha_dot = np.dot(alpha * y, X.dot(X.T).dot(alpha * y))
    return 0.5 * alpha_dot - alpha_sum

# Function for equality constraint: sum(alpha * y) = 0
def eq_constraint(alpha, y_train):
    return np.sum(alpha * y_train)

# Function for inequality constraints: 0 <= alpha_i <= C
def ineq_constraint(C, alpha):
    return C - alpha

def dual_svm(C_values, X_train, y_train):
    for C in C_values:
        # Initialize alpha values for optimization
        alpha_init = np.zeros(X_train.shape[0])

        # Set up bounds for alpha values (0 <= alpha_i <= C)
        bounds = [(0, C) for _ in range(len(alpha_init))]

        # Define equality constraint: sum(alpha * y) = 0
        cons = {'type': 'eq', 'fun': lambda alpha: eq_constraint(alpha, y_train)}

        # Define inequality constraints: 0 <= alpha_i <= C
        inequality_cons = {'type': 'ineq', 'fun': lambda alpha: ineq_constraint(alpha, C)}

        # Solve the optimization problem
        result = minimize(dual_objective, alpha_init, args=(X_train, y_train, C),
                        constraints=[cons, inequality_cons], bounds=bounds, method='SLSQP')

        # Retrieve optimized alphas
        alpha_optimized = result.x

        # Calculate bias term
        sv_indices = np.where(alpha_optimized > 1e-5)[0]
        support_vectors = X_train[sv_indices]
        support_labels = y_train[sv_indices]
        bias = np.mean(support_labels - np.dot(support_vectors, support_vectors.T.dot(alpha_optimized * support_labels)))

        # Calculate weights
        w = np.dot(X_train.T, alpha_optimized * y_train)

        # Print weights and bias
        print(f"For C={C}:")
        print(f"Weights (w): {w}")
        print(f"Bias (b): {bias}")

# Gaussian kernel function
def gaussian_kernel(X1, X2, gamma):
    pairwise_sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * pairwise_sq_dists)

# Dual SVM optimization with Gaussian Kernel
def fit(X, y, C, gamma):
    n_samples = X.shape[0]
    K = gaussian_kernel(X, X, gamma)
    P = np.outer(y, y) * K
    q = -np.ones(n_samples)
    G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
    h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * C))
    A = np.zeros((1, n_samples))
    A[0, :] = y
    b = np.array([0.0])

    # constrained optimization function that can handle convex problems and supports both dense and sparse matrices
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))

    alpha = np.array(sol['x']).flatten()
    return alpha

# Prediction function
def predict(X_train, y_train, alpha, X_test, gamma, bias=0.0):
    kernel = gaussian_kernel(X_train, X_test, gamma)
    predictions = np.dot((y_train * alpha), kernel) + bias
    return np.sign(predictions)

# Gaussian Kernal SVM
def gaussian_svm(gamma_values, C_values, X_train, X_test, y_train, y_test):
    best_error = float('inf')
    best_gamma = None
    best_C = None
    best_alpha = None
    best_bias = None
    
    for gamma in gamma_values:
        for C in C_values:
            # Train SVM with Gaussian Kernel
            alpha = fit(X_train, y_train, C, gamma)
            
            # Calculate bias term
            sv_indices = np.where(alpha > 1e-5)[0]
            alpha_sv = alpha[sv_indices]
            support_vectors = X_train[sv_indices]
            support_labels = y_train[sv_indices]
            kernel_sv = gaussian_kernel(support_vectors, support_vectors, gamma)
            # Calculate the bias
            bias = np.mean(support_labels - np.dot(kernel_sv.T, alpha_sv * support_labels))

            # Predict using the trained model
            y_pred_train = predict(X_train, y_train, alpha, X_train, gamma, bias)
            y_pred_test = predict(X_train, y_train, alpha, X_test, gamma, bias)

            # Calculate training and test errors
            train_error = np.mean(y_pred_train != y_train)
            test_error = np.mean(y_pred_test != y_test)

            # Print training and test errors for each combination
            print(f"For gamma={gamma}, C={C}:")
            print(f"Training Error: {train_error}, Test Error: {test_error}")

            # Check if this combination has the lowest test error
            if test_error < best_error:
                best_error = test_error
                best_gamma = gamma
                best_C = C
                best_alpha = alpha
                best_bias = bias

        # Print best combination        
    print(f"Best combination - Gamma: {best_gamma}, C: {best_C}")
    print(f"Best combination - Bias: {best_bias}")
