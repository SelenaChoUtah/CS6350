Adaboost and Bagging How-TO
Within the main file change either T of num_trees for the number of iterations you want to do
Comment in/out whichever function you want to run - super high-tech, I know


# Number of AdaBoost iterations
T = 500

# Call the Adaboost function
adaboost(btrain_data, btest_data, attributes, labels, T, weights)

# Number of Bagged Trees
num_trees = 500

 # Call the Bagged Trees function
test_predictions = bagged_trees(btrain_data, attributes, label, num_trees)
