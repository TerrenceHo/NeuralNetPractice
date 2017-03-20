import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def __init__(self):

def nnCostFunction(nn_params, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVar):

	# Flattens the Theta1, Theta2 from nn_params.
	# m = size(X,1)
	# Return Theta1Grad, Theta2Grad, and J the cost

	# create eye matrix
	# set y_matrix to y

	# feed forward through the neural network

	Theta1 = np.reshape(nn_params, (hiddenLayerSize, inputLayerSize+1))
	Theta2 = np.reshape(nn_params, (numLabels, hiddenLayerSize+1))
	m = X.shape[0]

	J = 0
	Theta1Grad = np.zeroes(np.shape(Theta1))
	Theta2Grad = np.zeroes(np.shape(Theta2))

	eye_matrix = np.eye(numLabels)
	y_matrix = eye_matrix(y,:)

	a1 = np.concatenate((np.ones((m,1)), X),1)
	z2 = a1 * Theta1.T
	a2 = np.concatenate((np.ones((m,1)), sigmoid(z2)),1)
	z3 = a2 * Theta.T
	a3 = sigmoid(z3)
	h = a3

	Theta1Reg = np.sum(np.sum(Theta1[:,1:]) ** 2)
	Theta2Reg = np.sum(np.sum(Theta2[:,1:]) ** 2)

	r = (lambdaVar/(2 * m)) * (Theta1Reg + Theta2Reg)

	J = (1/m) * np.sum(np.sum((-y_matrix) * log(h) - (1-y_matrix) * log(1-h))) + r

	d3 = a3 - y_matrix;
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


def sigmoid(z):

def sigmoidGradient(z):

def predict(theta1, theta2):