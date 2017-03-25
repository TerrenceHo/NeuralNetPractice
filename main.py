import neuralnet

# parameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

print("Loading Data")
datafile = 'data.mat'
mat = scipy.io.loadmat(datafile)
X, y = mat['X'], mat['y'] #load in x and y
X = np.insert(X, 0, 1 , axis = 1)
m = X[0].shape() # size of X



print("Placeholder for now, will show images and display data")

