from draw import Paint
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from neuralnet import forwardProp

def main():
    # initialize Paint class
    drawing = Paint()
    im_arr = drawing.get_digit()

    # retrieve weights from neuralnet.  Looks for 'Thetas.mat'
    data = loadmat('Thetas.mat')
    Theta1 = data['Theta1']
    Theta2 = data['Theta2']

    plt.axis('off')
    plt.imshow(im_arr, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

    im_vec = im_arr.reshape(1, -1)
    a1, z2, z2, z3, h = forwardProp(im_vec, Theta1, Theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    print(h)
    print(y_pred)

if __name__ == '__main__':
    main()
