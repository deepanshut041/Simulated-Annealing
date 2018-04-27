import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from network_helper import *
import sys
from random import random
from sa_helper import *


def l_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    costs = []

    # Parameters initialization.
    parameters = initialize_parameters_deep_sa(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
    
    return parameters


# Loading dataset
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]



# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

weight_single_cell = 4
print("Size required by one weight", weight_single_cell)
print ("train_x's shape: " + str(train_x.shape))
print("Total size of model will be require no weight in a layer * size required by one weight * Number of layers")
print("Let we have an input layer of 12288 and two output layer of 50 and 20 weights than total size size ", 12288 * 50 * 20 * weight_single_cell)

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288    # num_px * num_px * 3
n_h = int(n_x / 4)
n_y = 1
layers_dims = (n_x, 50, 20 , n_y)

def anneal():
    sol = (12288, 80, 50, 20, 1)
    parameters = l_layer_model(train_x, train_y, layers_dims, num_iterations=200, print_cost=True)
    old_cost = predict_accuracy(test_x, test_y, parameters)
    T = 1.0
    # Changes this for faster compilation 
    T_min = 0.00001
    alpha = 0.9
    costs = []
    while T > T_min:
        i = 1
        # Changes this for faster compilation 
        while i <= 10:
            new_parameters = l_layer_model(train_x, train_y, layers_dims, num_iterations=200, print_cost=True)
            new_cost = predict_accuracy(test_x, test_y, new_parameters)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random():
                parameters = new_parameters
                old_cost = new_cost
            i += 1
        T = T*alpha
        costs.append(new_cost)
        print(old_cost)
    return parameters, old_cost, costs

parameters, accuracy, costs = anneal()


print("Total size of model is equal to sum of size of its all parameters")
print("Size of input layer: ", sys.getsizeof(parameters['W1']))
print("Size of hidden layer 1: ", sys.getsizeof(parameters['W2']))
print("Size of layer hidden layer 2 : ", sys.getsizeof(parameters['W3']))
print("Total size : " , sys.getsizeof(parameters['W1']) + sys.getsizeof(parameters['W2']) + sys.getsizeof(parameters['W3']))

print("Calculated total size is : " , 12288 * 50 * 20 * weight_single_cell)
print("Total size is : " , 12288 * 50 * 20 * weight_single_cell)
print("Best accuracy is", accuracy)

plt.plot(costs)
plt.ylabel('Accuracy')
plt.xlabel('iletration')
plt.title("Simulated Annealing")
plt.show()