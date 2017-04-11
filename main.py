import numpy as np
import neuralnet as nn
import scipy.io



# parameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

print("Loading Data")
datafile = 'data.mat'
mat = scipy.io.loadmat(datafile)
X, y = mat['X'], mat['y'] #load in x and y
y[y==10] = 0 #set 10 in the dataset to 0
X = np.insert(X, 0, 1 , axis = 1) #add in a bias unit of 0
m = X[0].shape # size of X



print("Placeholder for now, will show images and display data")

#initializing random params for weights
initialTheta1 = nn.randInitWeights(input_layer_size, hidden_layer_size)
initialTheta2 = nn.reshapeParams(hidden_layer_size, num_labels)

#unroll initialTheta1/initialTheta2


