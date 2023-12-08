import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    # def __init__(self, input_size, hidden_layer_sizes, output_size):
    #     self.input_size = input_size
    #     self.hidden_layer_sizes = hidden_layer_sizes
    #     self.output_size = output_size
    #     self.num_layers = len(hidden_layer_sizes) + 1
        
    #     # Initialize weights and biases for each layer
    #     self.weights = [np.random.randn(input_size, hidden_layer_sizes[0])]
    #     self.biases = [np.random.randn(hidden_layer_sizes[0])]
    #     for i in range(len(hidden_layer_sizes) - 1):
    #         self.weights.append(np.random.randn(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
    #         self.biases.append(np.random.randn(hidden_layer_sizes[i+1]))
    #     self.weights.append(np.random.randn(hidden_layer_sizes[-1], output_size))
    #     self.biases.append(np.random.randn(output_size))

    # for zero weights
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_layer_sizes) + 1
        
        # Initialize weights and biases for each layer with zeros
        self.weights = [np.zeros((input_size, hidden_layer_sizes[0]))]
        self.biases = [np.zeros(hidden_layer_sizes[0])]
        for i in range(len(hidden_layer_sizes) - 1):
            self.weights.append(np.zeros((hidden_layer_sizes[i], hidden_layer_sizes[i+1])))
            self.biases.append(np.zeros(hidden_layer_sizes[i+1]))
        self.weights.append(np.zeros((hidden_layer_sizes[-1], output_size)))
        self.biases.append(np.zeros(output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def feedforward(self, x):
        # Feedforward propagation
        activations = [x]
        for i in range(self.num_layers):
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            activation = self.sigmoid(z)
            activations.append(activation)
        return activations
    
    def backpropagation(self, x, y):
        # Perform feedforward to get activations
        activations = self.feedforward(x)
        
        # Initialize delta values for gradients
        deltas = [None] * self.num_layers
        deltas[-1] = (activations[-1] - y) * self.sigmoid_derivative(activations[-1])
        
        # Backpropagate the errors
        for i in range(self.num_layers - 1, 0, -1):
            deltas[i - 1] = np.dot(deltas[i], self.weights[i].T) * self.sigmoid_derivative(activations[i])
        
        # Compute gradients of weights and biases
        batch_gradients_w = []
        batch_gradients_b = []
        for i in range(self.num_layers):
            batch_gradients_w.append(np.dot(activations[i].reshape(-1, 1), deltas[i].reshape(1, -1)))
            batch_gradients_b.append(deltas[i])
        
        return batch_gradients_w, batch_gradients_b
    
    def schedule_learning_rate(self, gamma_0, d, t):
        return gamma_0 / (1 + (gamma_0 / d) * t)
    
    def train_neural_network(self, X_train, y_train, X_test, y_test, hidden_layer_width):
        input_size = X_train.shape[1]
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1

        # Initialize the neural network with given hidden layer width
        nn = NeuralNetwork(input_size, [hidden_layer_width, hidden_layer_width], output_size)

        epochs = 100  # Number of epochs
        gamma_0 = 0.01  # Initial learning rate
        d = 500  # Decay rate parameter
        training_errors = []

        for epoch in range(epochs):
            # Shuffle training data at the start of each epoch
            shuffled_indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            for i, x in enumerate(X_train_shuffled):
                # Compute current learning rate based on schedule
                current_lr = self.schedule_learning_rate(gamma_0, d, epoch * len(X_train) + i)

                # Perform forward pass
                activations = nn.feedforward(x)

                # Compute loss and its derivative
                loss = 0.5 * np.square(activations[-1] - y_train_shuffled[i])
                loss_derivative = activations[-1] - y_train_shuffled[i]

                # Backpropagation
                deltas = [None] * nn.num_layers
                deltas[-1] = loss_derivative * nn.sigmoid_derivative(activations[-1])

                for j in range(nn.num_layers - 1, 0, -1):
                    deltas[j - 1] = np.dot(deltas[j], nn.weights[j].T) * nn.sigmoid_derivative(activations[j])

                # Compute gradients of weights and biases
                gradients_w = [np.dot(activations[j].reshape(-1, 1), deltas[j].reshape(1, -1)) for j in range(nn.num_layers)]
                gradients_b = deltas

                # Update weights and biases using SGD
                for j in range(nn.num_layers):
                    nn.weights[j] -= current_lr * gradients_w[j]
                    nn.biases[j] -= current_lr * gradients_b[j]

            # Calculate and print training error for this epoch
            train_predictions = np.array([nn.feedforward(x)[-1] for x in X_train])
            train_error = np.mean(0.5 * np.square(train_predictions - y_train))
            training_errors.append(train_error)
            # print(f"Epoch {epoch + 1}/{epochs} - Training error: {train_error}")

        plt.plot(range(epochs),training_errors, label= f'Width {hidden_layer_width}' )
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'Width {hidden_layer_width}: Training Error vs. Epochs')
        plt.legend()
        # plt.show()
        filename = 'c.png'  # Construct the filename using f-string
        plt.savefig(filename)   

        # Calculate and print test error after training
        test_predictions = np.array([nn.feedforward(x)[-1] for x in X_test])
        test_error = np.mean(0.5 * np.square(test_predictions - y_test))
        print(f"Width: {hidden_layer_width} - Training error: {train_error}")
        print(f"Test error: {test_error}")


# Define logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define objective function with Gaussian prior
def objective_function(X, y, w, v):
    N = len(y)
    h = sigmoid(np.dot(X, w))
    likelihood_term = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    prior_term = np.sum(w**2) / (2 * v)
    return likelihood_term + prior_term

# Calculate gradient of the objective function
def gradient(X, y, w, v):
    N = len(y)
    h = sigmoid(np.dot(X, w))
    grad_likelihood = np.dot(X.T, h - y) / N
    grad_prior = w / v
    return grad_likelihood + grad_prior

# Stochastic Gradient Descent (SGD)
def sgd(X_train, y_train, X_test, y_test, v, gamma_0, d, T):
    np.random.seed(42)
    w = np.zeros(X_train.shape[1])  # Initialize weights
    train_errors = []
    test_errors = []
    obj_values = []

    for epoch in range(1, T + 1):
        # Shuffle training data at the start of each epoch
        shuffle_idx = np.random.permutation(len(y_train))
        X_train_shuffled = X_train[shuffle_idx]
        y_train_shuffled = y_train[shuffle_idx]

        for i in range(len(y_train)):
            t = epoch * len(y_train) + i
            gamma_t = gamma_0 / (1 + gamma_0 / d * t)
            grad = gradient(X_train_shuffled[i:i+1], y_train_shuffled[i:i+1], w, v)
            w -= gamma_t * grad

        # Calculate training error
        train_pred = np.round(sigmoid(np.dot(X_train, w)))
        train_error = np.mean(train_pred != y_train)
        train_errors.append(train_error)

        # Calculate test error
        test_pred = np.round(sigmoid(np.dot(X_test, w)))
        test_error = np.mean(test_pred != y_test)
        test_errors.append(test_error)

        # Calculate objective function value
        obj_value = objective_function(X_train, y_train, w, v)
        obj_values.append(obj_value)

        # print(f"Epoch {epoch}/{T} - Training Error: {train_error}, Test Error: {test_error}, Objective Value: {obj_value}")
    

    return train_errors, test_errors, obj_values

        

