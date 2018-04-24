import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from network_helper import *
import sys

np.random.seed(1)

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):


    np.random.seed(1)
    costs = []

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward_sa(X, parameters)
        
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
# print ("test_x's shape: " + str(test_x.shape))

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288    # num_px * num_px * 3
n_h = int(n_x / 4)
n_y = 1
layers_dims = (n_x, 50, 20 , n_y)

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=200, print_cost=True)
print("Total size of model is equal to sum of size of its all parameters")
print("Size of input layer: ", sys.getsizeof(parameters['W1']))
print("Size of hidden layer 1: ", sys.getsizeof(parameters['W2']))
print("Size of layer hidden layer 2 : ", sys.getsizeof(parameters['W3']))
print("Total size : " , sys.getsizeof(parameters['W1']) + sys.getsizeof(parameters['W2']) + sys.getsizeof(parameters['W3']))

print("Calculated total size is : " , 12288 * 50 * 20 * weight_single_cell)
print("Total size is : " , 12288 * 50 * 20 * weight_single_cell)

pred_train = predict_sa(train_x, train_y, parameters)

pred_test = predict(test_x, test_y, parameters)
