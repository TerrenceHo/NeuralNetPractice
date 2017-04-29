# NeuralNetPractice
Neural Net that classifies digits, using a neural net with a single hidden
layer, with an accuracy rate about 95%.  Inspired by Andrew Ng's neural net from
Coursera.  Goes a step further by saving neural net weights, and allowing you to
use those weights to draw your own numbers and see if the neural net can predict
what you drew.  

neuralnet.py is the best neuralnet, it runs fastest using all matrix operations.
testneural.py also functions, bu runs with loops and is much slower.
testneural.py also doesn't save your weights.
predict.py allows you to use your saved weights to draw your own images and have
the neural net predict its value.  predict.py takes in a Thetas.mat, while
neuralnet.py generates a model.mat, so you must rename the file or else it won't
work.

Have fun!
