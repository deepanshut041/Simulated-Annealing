from random import random
import numpy as np

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from network_helper import *

np.random.seed(1)

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=300, print_cost=False):


    np.random.seed(1)
    costs = []

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

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
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    return parameters

def cost(sol):
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]


    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    layers_dims = sol

    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=200, print_cost=True)

    pred_test = predict_accuracy(test_x, test_y, parameters)

    print(pred_test)
    return pred_test


def acceptance_probability(old_cost, new_cost, T):
    a = 2.71828 ** ((new_cost - old_cost) / T)
    return a

def neighbor(tp):
    sol = list(tp)
    for i in range(1, len(sol)-1):
        if( np.random.rand() > .5):
            sol[i] = sol[i] + np.random.randint(0, 10)
        else :
            sol[i] = sol[i] - np.random.randint(0, 10)
    print(sol)

    return tuple(sol)