import numpy as np
import pandas as pd
from nnLib import *
import datetime
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__": 
    print('Start Time:', datetime.datetime.now())

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

    # Initialize the neural network
    input_size = X_train.shape[1]  # Number of features
    hidden_layer_sizes = [5, 4]   # Define the sizes of hidden layers
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1  # Assuming output size (number of classes)

    # Create the neural network
    nna = NeuralNetwork(input_size, hidden_layer_sizes, output_size)

    # Perform backpropagation for the first training example
    x = X_train[0]  # Use features of the first training example
    y = y_train[0]  # Use label of the first training example

    # Q2a) back-propagation
    batch_gradients_w, batch_gradients_b = nna.backpropagation(x, y)

    # Print the gradients for weights and biases
    for i in range(nna.num_layers):
        print(f"Gradients for weights at layer {i+1}:\n{batch_gradients_w[i]}")
        print(f"Gradients for biases at layer {i+1}:\n{batch_gradients_b[i]}")

    # Q2b/c) Implement sgd algo
    # Change the nnLib as needed for weights
    # Train neural network for different hidden layer widths
    hidden_layer_widths = [5, 10, 25, 50, 100]

    for width in hidden_layer_widths:
        nnb = NeuralNetwork(input_size, [width, width], output_size)
        nnb.train_neural_network(X_train, y_train, X_test, y_test, width)

    # Q2e) Pytorch
    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]
    activations = ['tanh', 'relu']

    # Convert data to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train.reshape(-1, 1))  # Reshape for compatibility
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test.reshape(-1, 1))

    for depth in depths:
        for width in widths:
            for activation in activations:
                class pyNeuralNetwork(nn.Module):
                    def __init__(self, input_size, output_size):
                        super(pyNeuralNetwork, self).__init__()
                        layers = []
                        layers.append(nn.Linear(input_size, width))
                        if activation == 'tanh':
                            nn.init.xavier_normal_(layers[0].weight)
                            layers.append(nn.Tanh())
                        elif activation == 'relu':
                            nn.init.kaiming_normal_(layers[0].weight, nonlinearity='relu')
                            layers.append(nn.ReLU())
                        for _ in range(depth - 2):
                            layers.append(nn.Linear(width, width))
                            if activation == 'tanh':
                                nn.init.xavier_normal_(layers[-1].weight)
                                layers.append(nn.Tanh())
                            elif activation == 'relu':
                                nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')
                                layers.append(nn.ReLU())
                        layers.append(nn.Linear(width, output_size))
                        self.model = nn.Sequential(*layers)

                    def forward(self, x):
                        return self.model(x)

                input_size = X_train.shape[1]
                output_size = 1  # Assuming regression problem, change for classification

                model = pyNeuralNetwork(input_size, output_size)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters())

                # Training loop
                epochs = 100
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_test_tensor)
                        val_loss = criterion(val_outputs, y_test_tensor)    
                # Test
                model.eval()
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)
                print(f"Depth: {depth}, Width: {width}, Activation: {activation}")
                print(f"Train Error: {val_loss.item()}")
                print(f"Test Error: {test_loss.item()}")
    
    # Q3a
    # Hyperparameters
    variance_values = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    gamma_0 = 0.01  # Initial learning rate
    d = 1000  # Decay rate
    T = 100  # Number of epochs

    # Perform hyperparameter tuning
    for v in variance_values:
        print(f"Training with variance = {v}")
        train_errors, test_errors, obj_values = sgd(X_train, y_train, X_test, y_test, v, gamma_0, d, T)
        
        # Plotting the objective function curve
        plt.plot(range(1, T + 1), obj_values, label=f"Variance = {v}")

        # Report training and test errors
        print(f"Variance = {v}: Final Training Error: {train_errors[-1]}, Final Test Error: {test_errors[-1]}")
        # plt.plot(train_errors, label= f'Variance: {v}' )
        # plt.xlabel('Epochs')
        # plt.ylabel('Error')
        # plt.title(f'Variance: {v}: Training Error vs. Epochs')
        # plt.legend()
        # plt.show()
    plt.xlabel('Epochs')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function Convergence')
    plt.legend()
    plt.show()
