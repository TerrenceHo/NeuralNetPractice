from draw import Paint
import numpy as np
import matplotlib.pyplot as plt

drawing = Paint()
im_arr = drawing.get_digit()

plt.axis('off')
plt.imshow(im_arr, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

im_vec = im_arr.reshape(1, -1)


