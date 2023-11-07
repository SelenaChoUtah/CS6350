# Perceptron Functions

Following three types of perceptron contains a training and testing function.

The only variable that requires changing will be the epochs.

If you want to use this library: **from perceptron import**

**Standard Perceptron**

perceptron_train(X_train, y_train, max_epochs=10)

perceptron_test(X_test, y_test, weights, bias)

**Voted Perceptron**

voted_perceptron_train(X_train, y_train, max_epochs=10)

voted_perceptron_test(X_test, y_test, weights_list, counts)

**Average Perceptron**

average_perceptron_train(X_train, y_train, max_epochs=10)

average_perceptron_test(X_test, y_test, avg_weight)
