import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def resample(data, n_samples=None):
    if n_samples is None:
        n_samples = len(data)

    # Generate random indices with replacement
    indices = np.random.choice(len(data), n_samples, replace=True)

    # Select the samples based on the random indices
    resampled_data = data.iloc[indices]

    return resampled_data


def entropy(data,labels):
    label_counts = data['label'].value_counts()
    total_samples = len(data)
    entropy_value = 0
    
    for label in label_counts.index:  # Iterate over the actual label values
        probability = label_counts[label] / total_samples
        entropy_value -= probability * np.log2(probability)
    return entropy_value

# Function to compute information gain for a given feature
def information_gain(data, feature, labels):
    total_entropy = entropy(data, labels)  # Compute entropy based on labels directly
    values = data[feature].unique()
    weighted_entropy = 0

    for value in values:
        subset = data[data[feature] == value]
        weighted_entropy += len(subset) / len(data) * entropy(subset, labels)  # Compute subset entropy based on labels

    return total_entropy - weighted_entropy

# Function to train a decision stump
def train_decision_stump(data, attributes, labels, weights):
    best_attribute = None
    best_threshold = None
    best_error = float('inf')
    best_predictions = None

    for attribute in attributes:
        unique_values = data[attribute].unique()

        for value in unique_values:
            predictions = np.where(data[attribute] <= value, 'yes', 'no')
            error = (weights * (predictions != data['label'])).sum()

            if error < best_error:
                best_error = error
                best_attribute = attribute
                best_threshold = value
                best_predictions = predictions

    # Return default predictions if no valid split is found
    if best_predictions is None:
        default_prediction = np.full(len(data), 'no')  # Replace 'no' with the desired default prediction
        return best_attribute, best_threshold, default_prediction, best_error

    return best_attribute, best_threshold, best_predictions, best_error

def id3(data, attributes, labels, max_depth, measure):
    if len(labels) == 1:
        return labels[0]
    if len(attributes) == 0 or max_depth == 0:
        return data['label'].mode().iloc[0]

    best_attribute = None
    best_value = None
    best_measure_value = None

    for attribute in attributes:
        measure_value = measure(data, attribute, labels)
        if best_attribute is None or measure_value < best_measure_value:
            best_attribute = attribute
            best_measure_value = measure_value

    tree = {best_attribute: {}}

    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        subtree = id3(subset, [attr for attr in attributes if attr != best_attribute], labels, max_depth - 1, measure)
        tree[best_attribute][value] = subtree

    return tree

def adaboost(btrain_data, btest_data, attributes, labels, T, weights):

    # Lists to store errors
    train_errors = []
    test_errors = []
    stump_errors = []

    ada_predictions_train = np.zeros(len(btrain_data))
    ada_predictions_test = np.zeros(len(btest_data))

    for t in range(T):
        print(f"Iteration {t + 1}")

        # Train a decision stump
        best_attribute, best_threshold, train_predictions, stump_error = train_decision_stump(
            btrain_data, attributes, labels, weights )

        # Calculate alpha (weight) for the stump
        alpha = 0.5 * np.log((1 - stump_error) / max(stump_error, 1e-10))

        # Update example weights
        weights = weights * np.exp(-alpha * (train_predictions != btrain_data['label']))

        # Normalize weights
        weights /= weights.sum()

        # Append stump error to the list
        stump_errors.append(stump_error)

        # Calculate AdaBoost prediction for training and test datasets
        ada_predictions_train += alpha * (train_predictions == 'yes')
        ada_predictions_test += alpha * (btest_data[best_attribute] == best_threshold)

        # Calculate training and test errors based on AdaBoost predictions
        train_error = 1 - (
            np.sum(np.sign(ada_predictions_train) == (btrain_data['label'] == 'yes'))
            / len(btrain_data)
        )
        test_error = 1 - (
            np.sum(np.sign(ada_predictions_test) == (btest_data['label'] == 'yes'))
            / len(btest_data)
        )

        # Append errors to the lists
        train_errors.append(train_error)
        test_errors.append(test_error)

    # Plot the errors outside of the loop
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(range(1, T + 1), train_errors, label='Training Error')
    plt.plot(range(1, T + 1), test_errors, label='Test Error')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Errors')
    plt.legend()

    plt.subplot(122)
    plt.plot(range(1, T + 1), stump_errors, label='Stump Error')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Stump Error')
    plt.title('Stump Errors')

    plt.tight_layout()
    plt.savefig("adaboost_errors.png")
    plt.show()
    print(train_errors)
    print(test_errors)
    print(stump_errors)


def bagged_trees(btrain_data, attributes, labels, num_trees):
    bagged_tree_ensemble = []

    train_errors = []  # List to store training errors for each iteration
    test_errors = []   # List to store test errors for each iteration

    for _ in range(num_trees):
        # Step 1: Resample the training data with replacement
        bootstrap_sample = resample(btrain_data)

        # Step 2: Train a decision tree on the bootstrap sample
        tree = id3(bootstrap_sample, attributes, labels, max_depth=3, measure=information_gain)

        # Add the trained tree to the ensemble
        bagged_tree_ensemble.append(tree)

        # Calculate training and test errors for this iteration
        train_error = compute_error(btrain_data, bagged_tree_ensemble, attributes, labels)
        test_error = compute_error(btest_data, bagged_tree_ensemble, attributes, labels)

        train_errors.append(train_error)
        test_errors.append(test_error)

    # Plot the training and test errors
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_trees + 1), train_errors, label='Training Error')
    plt.plot(range(1, num_trees + 1), test_errors, label='Test Error')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Errors for Bagging')
    plt.legend()
    plt.savefig("bagging_errors.png")
    plt.show()

def compute_error(data, ensemble, attributes, labels):
    errors = 0

    for index, test_example in data.iterrows():
        # Use majority voting to determine the final prediction
        predictions = [predict(tree, test_example) for tree in ensemble]
        final_prediction = max(set(predictions), key=predictions.count)

        if final_prediction != test_example['label']:
            errors += 1

    error_rate = errors / len(data)
    return error_rate
            
def predict(tree, example):
    # Recursive function to make predictions using a decision tree
    for attribute, subtree in tree.items():
        value = example[attribute]
        if value in subtree:
            if isinstance(subtree[value], dict):
                return predict(subtree[value], example)
            else:
                return subtree[value]
            
            
# Define the cost function (Mean Squared Error)
def cost_function(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)
    error = predictions - y
    return (1 / (2 * m)) * np.sum(error**2)

def lms(ctrain_data, ctest_data, learning_rate, tolerance, max_iterations):
    # Preprocess the data
    # Add a bias term (intercept) to the features
    ctrain_data['bias'] = 1
    ctest_data['bias'] = 1

    # Separate features (X) and target (y)
    X_train = ctrain_data.iloc[:, :-1]  
    y_train = ctrain_data.iloc[:, -1]  

    X_test = ctest_data.iloc[:, :-1]  
    y_test = ctest_data.iloc[:, -1]  

    # Initialize weights
    num_features = X_train.shape[1]
    weights = np.array([random.uniform(-0.1, 0.1) for _ in range(num_features)])

    # Initialize variables to store cost function values and weights
    train_cost_history = []  # To store training costs
    test_cost_history = []   # To store testing costs

    # Gradient Descent Loop
    for iteration in range(max_iterations):
        
        # Calculate the gradient
        predictions = X_train.dot(weights)
        gradient = (1 / len(y_train)) * X_train.T.dot(predictions - y_train)

        # Update weights
        learning_rate = learning_rate / (1 + iteration * 0.001)
        weights -= learning_rate * gradient

        # Calculate the cost and add it to the history
        train_cost = cost_function(X_train, y_train, weights)
        train_cost_history.append(train_cost)
        
        test_cost = cost_function(X_test, y_test, weights)
        test_cost_history.append(test_cost)

        # # Check for convergence
        # if iteration > 0 and abs(cost_history[-2] - cost_history[-1]) < tolerance:
        #     break

    # Testing: Calculate the cost function on the test data
    # train_cost = cost_function(X_train, y_train, weights)
    # test_cost = cost_function(X_test, y_test, weights)

    # Plot the learning curves for training and testing
    plt.plot(range(max_iterations), train_cost_history, label='Training Cost')
    plt.plot(range(max_iterations), test_cost_history, label='Testing Cost')
    plt.xlabel("Iterations")
    plt.ylabel("Cost Function")
    plt.title("Learning Curves")
    plt.legend()
    plt.savefig("lms.png")
    plt.show()

    # Report the results
    print("Learned weights:", weights)
    print("Learning rate:", learning_rate)
    print("Final training cost:", train_cost_history[-1])
    print("Final training cost:", test_cost_history[-1])
    print("Test cost:", test_cost)
    
    return test_cost_history

# Stochastic Gradient Descent (SGD) function
def sgd(ctrain_data, ctest_data, learning_rate, max_iterations):

    # Separate features (X) and target (y)
    X_train = ctrain_data.iloc[:, :-1]  
    y_train = ctrain_data.iloc[:, -1]  

    X_test = ctest_data.iloc[:, :-1]  
    y_test = ctest_data.iloc[:, -1] 

    num_features = X_train.shape[1]
    weights = np.zeros(num_features)
    train_cost_history = [] 
    
    for iteration in range(max_iterations):
        for i in range(X_train.shape[0]):
            random_index = np.random.randint(0, X_train.shape[0])
            xi = X_train.iloc[random_index]
            yi = y_train.iloc[random_index]
            prediction = np.dot(xi, weights)
            error = prediction - yi
            gradient = xi * error
            weights = weights - learning_rate * gradient

        train_cost = cost_function(X_train, y_train, weights)
        train_cost_history.append(train_cost)
    
    # Calculate the cost function on the test data with the learned weights
    test_cost = cost_function(X_test, y_test, weights)

    # Plot the learning curve
    plt.plot(range(len(train_cost_history)), train_cost_history)
    plt.xlabel("Number of Updates")
    plt.ylabel("Training Cost")
    plt.title("Learning Curve (Training Cost)")
    plt.savefig("sgd.png")
    plt.show()

    # Report the results
    print("Learned weights:", weights)
    print("Learning rate:", learning_rate)
    print("Final training cost:", train_cost_history[-1])
    print("Test cost:", test_cost)

    return weights, train_cost_history, test_cost

# optimal weight vector
def optimal_solution(ctrain_data):
    # Separate features (X) and target (y)
    X_train = ctrain_data.iloc[:, :-1]  
    y_train = ctrain_data.iloc[:, -1] 

    XTX = np.dot(X_train.T, X_train)
    XTy = np.dot(X_train.T, y_train)
    optimal_weights = np.linalg.solve(XTX, XTy)
    return optimal_weights
            

if __name__ == "__main__": 
    # Load training and test data
    column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    btrain_data = pd.read_csv("btrain.csv",names=column_names)
    btest_data = pd.read_csv("btest.csv",names=column_names)
    
    # Define features and labels
    label =  ['no','yes']
    attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

    # Initialize weights for training examples
    weights = np.ones(len(btrain_data)) / len(btrain_data)

    # Number of AdaBoost iterations
    T = 500

    adaboost(btrain_data, btest_data, attributes, labels, T, weights)

    # Number of Bagged Trees
    num_trees = 500

     # Call the Bagged Trees function
    test_predictions = bagged_trees(btrain_data, attributes, label, num_trees)

    # Calculate the test error by comparing the predictions to the actual labels
    test_error = 1 - (test_predictions == btest_data['label']).mean()
    print(f"Test Error: {test_error}")

    # Concrete Data
    column_names = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr','SLUMP'
]
    ctrain_data = pd.read_csv("ctrain.csv",names=column_names)
    ctest_data = pd.read_csv("ctest.csv",names=column_names)

    # Preprocess the data
    # Add a bias term (intercept) to the features
    ctrain_data['bias'] = 1
    ctest_data['bias'] = 1

    # Set hyperparameters
    learning_rate = 0.0001
    tolerance = 1e-6
    max_iterations = 1000

    lms(ctrain_data, ctest_data, learning_rate, tolerance, max_iterations)

    learned_weights, train_cost_history, test_cost = sgd(ctrain_data, ctest_data, learning_rate, max_iterations)

    optimal_weights = optimal_solution(ctrain_data)
    print(optimal_weights)
