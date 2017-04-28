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


if __name__ == '__main__':
    main()
