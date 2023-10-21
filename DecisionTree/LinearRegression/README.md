The goal of this linear regression file is how to set the parameters. 
Change the following hyperparameter within the main file to whatever you desired
Remember to comment in/out the function you want to run

# Set hyperparameters
learning_rate = 0.0001
tolerance = 1e-6
max_iterations = 1000

# Least-Means Squared method
lms(ctrain_data, ctest_data, learning_rate, tolerance, max_iterations)

# Stochastic Gradient Descent Algorithm
learned_weights, train_cost_history, test_cost = sgd(ctrain_data, ctest_data, learning_rate, max_iterations)

# Identifying optimal weights
optimal_weights = optimal_solution(ctrain_data)
