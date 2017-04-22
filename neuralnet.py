import numpy as np
from scipy.io import loadmat
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

#Global Variables
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
iterations = 250

def main():
    print("Loading Data")
    datafile = 'data.mat'
    mat = loadmat(datafile)
    X, y = mat['X'], mat['y']
    y[y==10] = 0
    m = X[0].shape
    encoder = OneHotEncoder(sparse = False)
    y_onehot = encoder.fit_transform(y)

    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels
                               * (hidden_size + 1)) - 0.5) * 0.25
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)],
                                  (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):],
                                  (num_labels, (hidden_size + 1))))

    print("Start Neural Net Training")
    input("Press Enter to continue...")
    fmin = minimize(fun=nnCostFunction, x0=params, args=(input_size, hidden_size,
        num_labels, X, y_onehot, learning_rate), method='TNC', jac=True,
        options={'maxiter': iterations})
    print(fmin)

    theta1 = np.reshape(fmin.x[:hidden_size * (input_size + 1)],
                                  (hidden_size, (input_size + 1)))
    theta2 = np.reshape(fmin.x[hidden_size * (input_size + 1):],
                                  (num_labels, (hidden_size + 1)))

    print("Predict Accuracy")
    input("Press Enter to continue...")
    a1, z2, a2, z3, h = forwardProp(X, theta1, theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('Accuracy = {0}%'.format(accuracy * 100))

def nnCostFunction(params, input_size, hidden_size, num_labels, X, y,
                   learning_rate):
    #reshape parameters
    Theta1 = np.reshape(params[:hidden_size * (input_size + 1)],
                                  (hidden_size, (input_size + 1)))
    Theta2 = np.reshape(params[hidden_size * (input_size + 1):],
                                  (num_labels, (hidden_size + 1)))
    m = X.shape[0]

    # Feed Forward Network
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1.dot(Theta1.T)
    a2 = np.insert(expit(z2), 0, values=np.ones(m), axis=1)
    z3 = a2.dot(Theta2.T)
    a3 = expit(z3)
    h = a3

    Theta1Reg = np.sum(np.sum(Theta1[:,1:]) ** 2)
    Theta2Reg = np.sum(np.sum(Theta2[:,1:]) ** 2)

    r = (learning_rate/(2 * m)) * (Theta1Reg + Theta2Reg)

    J = (1/m) * np.sum(np.sum((-y) * np.log(h) - (1-y) * np.log(1-h))) + r

    d3 = a3 - y
    d2 = sigmoidGradient(z2) * (d3.dot(Theta2[:,1:]))

    Delta1 = d2.T.dot(a1)
    Delta2 = d3.T.dot(a2)

    # computing weights from backprob derivatives
    Theta1Grad = (1/m) * Delta1
    Theta2Grad = (1/m) * Delta2

    # regularizing the backprop
    Theta1[:,0] = 0
    Theta2[:,0] = 0

    # Getting new weights
    Theta1Grad = Theta1Grad + ((learning_rate/m) * Theta1)
    Theta2Grad = Theta2Grad + ((learning_rate/m) * Theta2)
    grad = np.concatenate((np.ravel(Theta1Grad), np.ravel(Theta2)))

    # Return both the weights and the cost
    return J, grad

def forwardProp(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1.dot(theta1.T)
    a2 = np.insert(expit(z2), 0, values=np.ones(m), axis=1)
    z3 = a2.dot(theta2.T)
    h = expit(z3)
    return a1, z2, a2, z3, h

def sigmoidGradient(z):
    d = expit(z)
    return d*(1-d)

if __name__ == '__main__':
    main()
