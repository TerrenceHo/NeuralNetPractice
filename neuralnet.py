import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import scipy.misc
import scipy.optimize
from scipy.special import expit
import matplotlib.cm as cm
import random
import itertools


def nnCostFunction(nn_params, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVar):

    # Flattens the Theta1, Theta2 from nn_params.
    # m = size(X,1)
    # Return Theta1Grad, Theta2Grad, and J the cost

    # create eye matrix
    # set y to y

    # feed forward through the neural network

    Theta1 = np.reshape(nn_params, (hiddenLayerSize, inputLayerSize+1))
    Theta2 = np.reshape(nn_params, (numLabels, hiddenLayerSize+1))
    m = X.shape[0]

    J = 0
    Theta1Grad = np.zeros(np.shape(Theta1))
    Theta2Grad = np.zeros(np.shape(Theta2))

    rows, cols = y.shape
    y1 = np.zeros((rows, numLabels))
    for i in range(0, rows):
        y1[i, y[i]] = 1

    a1 = np.concatenate((np.ones((m,1)), X),1)
    z2 = a1 * Theta1.T
    a2 = np.concatenate((np.ones((m,1)), sigmoid(z2)),1)
    z3 = a2 * Theta2.T
    a3 = sigmoid(z3)
    h = a3

    Theta1Reg = np.sum(np.sum(Theta1[:,1:]) ** 2)
    Theta2Reg = np.sum(np.sum(Theta2[:,1:]) ** 2)

def backPropagate(mythetas_flat, myx_flat, mmy, lambda):
    r = (lambdaVar/(2 * m)) * (Theta1Reg + Theta2Reg)

    J = (1/m) * np.sum(np.sum((-y1) * log(h) - (1-y1) * log(1-h))) + r

    d3 = a3 - y1;
    d2 = sigmoidGradient(z2) * (d3 * Theta2[:,1:])

    Delta1 = d2.T * a1
    Delta2 = d3.T * a2

    Theta1Grad = (1/m) * Delta1
    Theta2Grad = (1/m) * Delta2


    #CHECK
    Theta1[:,0] = 0
    Theta2[:,0] = 0

    Theta1Grad = Theta1Grad + ((lambdaVar/m) * Theta1)
    Theta2Grad = Theta2Grad * ((lambdaVar/m) * Theta2)

def randInitWeights(layerIn, layerOut):
    epsilon_init = 0.12
    return rand(layerIn, 1 + layerIn) * 2 * episilon_init - epsilon_init

def sigmoid(z):
    g = 1.0/(1.0 + exp(z))
    return g
def sigmoidGradient(z):
    return sigmoid(z) * (1-sigmoid(z))

def predict(theta1, theta2):
    print("Predict")

# Helper Functions.  These all flatten various thetas
def flattenParams(thetas_list):
    # Hand this function a list of theta matrices, and it will flatten it into one long (n,1) shaped numpy array

    flattened_list = [ mytheta.flatten() for mytheta in thetas_list ]
    combined = list(itertools.chain.from_iterable(flattened_list))
    assert len(combined) == (input_layer_size+1)*hidden_layer_size + (hidden_layer_size+1)*output_layer_size
    return np.array(combined).reshape((len(combined),1))

def reshapeParams(flattened_array):
    theta1 = flattened_array[:(input_layer_size+1)*hidden_layer_size].reshape((hidden_layer_size,input_layer_size+1))
    theta2 = flattened_array[(input_layer_size+1)*hidden_layer_size:].reshape((output_layer_size,hidden_layer_size+1))
    return [ theta1, theta2 ]

def flattenX(myX):
    return np.array(myX.flatten()).reshape((n_training_samples*(input_layer_size+1),1))

def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((n_training_samples,input_layer_size+1))


