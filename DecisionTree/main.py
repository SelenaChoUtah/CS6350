import numpy as np
import pandas as pd
import datetime
from Perceptron import *

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

    # Augment for bias term
    X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))
    X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))

    # print(X_train)

    # Make sure the labels are in the format of -1 (for forged) and 1 (for genuine)
    # Output is a label y ∈ {-1, 1}
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # Standard Perceptron
    learned_weights, learned_bias, training_errors = perceptron_train(X_train, y_train, max_epochs=10)

    # Test the Perceptron on the test dataset
    test_error = perceptron_test(X_test, y_test, learned_weights, learned_bias)

    print("Standard Perceptron: Learned Weight Vector:", learned_weights)
    print("Standard Perceptron: Average Prediction Error on Test Dataset:", test_error)
    print("Standard Perceptron: Average Prediction Accuracy on Test Dataset:", 1-test_error)

    # Voted Perceptron
    weights_list, counts = voted_perceptron_train(X_train, y_train, max_epochs=10)

    # Test the Voted Perceptron on the test dataset
    test_error = voted_perceptron_test(X_test, y_test, weights_list, counts)

    # Print the list of distinct weight vectors and their counts
    for i, (w, count) in enumerate(zip(weights_list, counts)):
        print(f"Weight Vector {i + 1}: {np.round(w,3)}, Count: {count}")

    print("Voted Perceptron: Average Test Error:", test_error)
    print("Voted Perceptron: Average Accuracy Error:", 1-test_error)

    # Average Perceptron
    avg_weight = average_perceptron_train(X_train, y_train, max_epochs=10)
    avg_error = average_perceptron_test(X_test, y_test, avg_weight)
    print("Average Perceptron: Weight Vector: ",avg_weight)
    print("Average Perceptron: Average Prediction Error: ", avg_error)
