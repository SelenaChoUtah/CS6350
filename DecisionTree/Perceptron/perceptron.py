import numpy as np
import pandas as pd

def perceptron_train(X_train, y_train, max_epochs=10):
    # Initialize weights to zeros
    num_features = X_train.shape[1]
    weights = np.zeros(num_features)
    bias = 0
    learning_rate = 0.1
    errors = []

    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(X_train)):
            xi = X_train[i]
            yi = y_train[i]

            # Calculate the prediction using the current weights
            prediction = np.dot(xi, weights) + bias

            # Update the weights if there is a misclassification
            if yi * prediction <= 0:
                weights += learning_rate * yi * xi
                bias += learning_rate * yi
                total_error += 1

        errors.append(total_error)
        if total_error == 0:
            break

    return weights, bias, errors

def perceptron_test(X_test, y_test, weights, bias):
    num_samples = X_test.shape[0]
    num_errors = 0

    for i in range(num_samples):
        xi = X_test[i]
        yi = y_test[i]

        # Calculate the prediction using the learned weights
        prediction = np.dot(xi, weights) + bias

        if yi * prediction <= 0:
            num_errors += 1

    return num_errors / num_samples

def voted_perceptron_train(X_train, y_train, max_epochs=10):
    num_samples, num_features = X_train.shape
    # Initialize
    weights_list = []  # List to store weight vectors
    counts = []        # List to store counts for each weight vector

    # Initialize the first weight vector to zeros
    w = np.zeros(num_features)

    for epoch in range(max_epochs):
        for i in range(num_samples):
            xi = X_train[i]
            yi = y_train[i]

            # Calculate the prediction using the current weight vector
            prediction = np.sign(np.dot(xi, w))

            if prediction != yi:
                weights_list.append(w.copy())
                counts.append(1)
                w = w + yi * xi
            else:
                counts[-1] += 1

    # Add the final weight vector to the list
    weights_list.append(w)
    counts.append(1)

    return weights_list, counts

def voted_perceptron_test(X_test, y_test, weights_list, counts):
    num_samples = X_test.shape[0]
    num_weights = len(weights_list)
    num_errors = 0

    for i in range(num_samples):
        xi = X_test[i]
        yi = y_test[i]

        weighted_predictions = [np.sign(np.dot(xi, w)) for w in weights_list]

        # Make a weighted prediction based on the counts
        weighted_prediction = np.sign(np.sum([counts[j] * weighted_predictions[j] for j in range(num_weights)]))

        if weighted_prediction != yi:
            num_errors += 1

    return num_errors / num_samples

def average_perceptron_train(X_train, y_train, max_epochs=10):
    num_samples, num_features = X_train.shape
    # Initialize the weight vector to zeros
    w = np.zeros(num_features)

    # Initialize the total weight to zeros
    total_weight = np.zeros(num_features)

    for epoch in range(max_epochs):
        for i in range(num_samples):
            xi = X_train[i]
            yi = y_train[i]

            # Calculate the prediction using the current weight vector
            prediction = np.sign(np.dot(xi, w))

            if prediction != yi:
                w = w + yi * xi

        # Accumulate the weight vector at the end of each epoch
        total_weight += w

    # Calculate the average weight vector
    avg_weight = total_weight / (max_epochs * num_samples)

    return avg_weight

def average_perceptron_test(X_test, y_test, avg_weight):
    num_samples = X_test.shape[0]
    num_errors = 0

    for i in range(num_samples):
        xi = X_test[i]
        yi = y_test[i]

        # Make a prediction using the average weight vector
        prediction = np.sign(np.dot(avg_weight, xi))

        if prediction != yi:
            num_errors += 1

    return num_errors / num_samples
