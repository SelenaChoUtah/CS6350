import numpy as np
import pandas as pd
import datetime
from svm import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize


if __name__ == "__main__": 
    print('Start Time: ',datetime.datetime.now())

    # Load the training and test data
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    # Extract features (X) and labels (y) from the data
    X_train = train_data.iloc[:, :-1].values  # Features from all but the last column
    y_train = train_data.iloc[:, -1].values   # Labels from the last column

    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Preprocess Data
    y_train = np.where(y_train == 0, -1, 1)  # Convert labels to {1, -1}
    y_test = np.where(y_test == 0, -1, 1)
    # print(X_train)

    C_values = [100 / 873, 500 / 873, 700 / 873]
    gamma_0_values = [0.1,0.01, 0.001]  # tune for convergence
    a_values = [1, 0.1, 0.01]  # tune for convergence
    T = 100  # Maximum epochs

    # Problem 2
    tra, tea, wa, ba = schedule2A(C_values, gamma_0_values, a_values, T, X_train, y_train, X_test, y_test)
    trb, teb, wb, bb = schedule2B(C_values, gamma_0_values, a_values, T, X_train, y_train, X_test, y_test)

    print(np.array(tra)-np.array(trb))
    print(np.array(tea)-np.array(teb))
    print(np.array(ba) - np.array(bb))

    # Problem 3A: Dual SVM Learning
    dual_svm(C_values, X_train, y_train)

    # Problem 3B: Gaussian Kernal in Dual SVM Learning
    # Test different values of gamma and C    
    gamma_values = [0.1, 0.5, 1, 5, 100]
    C_values = [100 / 873, 500 / 873, 700 / 873]

    gaussian_svm(gamma_values, C_values, X_train, X_test, y_train, y_test)

    # Problem 3C: Report number of overlapped support vectors
    # support_vectors_per_gamma = {}  # Dictionary to store support vectors for each gamma
    C_value = 500 / 873
    support_vectors_per_gamma = {}  # Dictionary to store support vectors for each gamma

    for gamma in gamma_values:
        for C in C_values:
            # Train SVM with Gaussian Kernel
            alpha = fit(X_train, y_train, C, gamma)
            
            # Find support vectors
            sv_indices = np.where(alpha > 1e-5)[0]
            support_vectors = X_train[sv_indices]
            support_vectors_per_gamma[gamma] = support_vectors
            print(f"For gamma={gamma}, C={C}: Number of support vectors = {len(support_vectors)}")

    # Find overlapped support vectors between consecutive gamma values
    overlapped_support_vectors = {}
    prev_sv = None

    for gamma, sv in support_vectors_per_gamma.items():
        if prev_sv is not None:
            overlapped = np.intersect1d(prev_sv, sv, assume_unique=True)
            overlapped_support_vectors[(prev_gamma, gamma)] = len(overlapped)
        
        prev_gamma = gamma
        prev_sv = sv
        
    # Report the number of overlapped support vectors
    for gammas, count in overlapped_support_vectors.items():
        print(f"Overlapped support vectors between gamma={gammas[0]} and gamma={gammas[1]}: {int(np.round(count/5))}")

   
