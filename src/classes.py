import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import sys

class ImportData(object):
    """
    Import datasets with different formats
    Imput: the list of possible labels
    Return: (X, y)
    """
    def change_y_shape(self, y):
        return np.array([np.argmax(x) for x in y]).reshape((len(y), 1)).astype(np.int)

    def digitsImport(self, filename):
        """
        Split with ' '
        Attribute a class in function of the column with 1
        Divise features by n
        """
        f = open(filename, 'r')
        M = np.array([line.strip().split(' ') for line in f])
        X = scale(M[:,0:-10].astype(np.float))
        y = M[:,-10:].astype(np.int)
        return (X, y)

    def irisImport(self, filename, n):
        """
        Split with ','
        Scale all features (N(0,1))
        Divise features by n
        """
        f = open(filename, 'r')
        M = np.array([line.strip().split(',') for line in f])
        X = M[:,:-1].astype(np.float)
        divisor = np.empty(X.shape)
        divisor.fill(float(n))
        X = scale(np.divide(X, divisor))
        y = M[:,-1].astype(int)
        new_y = np.zeros((len(y), 3))
        for i in range(len(y)):
            new_y[i, y[i]] = 1
        return (X, new_y.astype(np.float))

    def breastCancerImport(self, filename, n):
        """
        Split with ','
        Remove lines with '?' in values
        Do not use the id in first column
        Scale all features (N(0,1))
        Divise features by n
        """
        f = open(filename, 'r')
        M = np.array([line.strip().split(',') for line in f if ('?' not in line)])
        X = M[:,1:-1].astype(np.float)
        divisor = np.empty(X.shape)
        divisor.fill(float(n))
        X = scale(np.divide(X, divisor))
        y = []
        for i in M[:,-1].astype(int):
            y.append([1, 0] if i==2 else [0, 1])
        return (X, np.array(y).reshape(len(y), 2))


class NeuralNetwork(object):
    def __init__(self, num_input, num_hidden, num_output=10, learning_rate=1, momentum=1.0, num_iter=100):
        """ Initialize the NeuralNetwork class with parameters """
        self.input_nb = num_input
        self.hidden_nb = num_hidden
        self.output_nb = num_output
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_iter = num_iter
        self.temp_in = np.zeros((self.input_nb, self.hidden_nb))
        self.temp_out = np.zeros((self.hidden_nb, self.output_nb))
        self.activation_input = [0.0 for i in range(self.input_nb)]
        self.activation_hidden = [0.0 for i in range(self.hidden_nb)]
        self.activation_output = [0.0 for i in range(self.output_nb)]
        self.wi = np.random.normal(0.0, 1.0, (self.input_nb, self.hidden_nb))
        self.vi = np.random.normal(0.0, 1.0, (self.hidden_nb, self.output_nb))

    def tanh(self, x):
        """ Compute the tangante of x """
        return math.tanh(x)

    def derivated_tanh(self, x):
        """ Compute the derivated tangeante_h of x """
        return float(1.0 - x * x)
    def sigmoid(self, x):
        """ Compute the sigmoid of x """
        return float(1.0 / (1.0 + np.exp(-x)))
    def derivated_sigmoid(self, x):
        """ Compute the derivated sigmoid of x """
        return float(x * (1.0 - x))

    def propagation(self, X):
        """ Compute all coefficients for each layer, until the output """
        self.activation_input = X

        for j in range(self.hidden_nb):
            total = np.sum([self.activation_input[i] * self.wi[i][j] for i in range(self.input_nb)])
            self.activation_hidden[j] = self.tanh(total)

        for k in range(self.output_nb):
            total = np.sum([self.activation_hidden[j] * self.vi[j][k] for j in range(self.hidden_nb)])
            self.activation_output[k] = self.sigmoid(total)

        return self.activation_output

    def compute_errors(self, y):
        """ Compute errors between layers """
        output_error = [0.0 for i in range(self.output_nb)]
        for k in range(self.output_nb):
            error = - (y[k] - self.activation_output[k])
            output_error[k] = self.derivated_sigmoid(self.activation_output[k]) * error

        hidden_errors = [0.0 for i in range(self.hidden_nb)]
        for j in range(self.hidden_nb):
            error = sum([output_error[k] * self.vi[j][k] for k in range(self.output_nb)])
            hidden_errors[j] = self.derivated_tanh(self.activation_hidden[j]) * error

        return (output_error, hidden_errors)

    def train(self, X, y):
        for i in range(self.num_iter):
            for index in range(X.shape[0]):
                self.propagation(X[index])
                self.back_propagation(y[index])

    def train_unique(self, X, nb_class=3):
        for i in range(self.num_iter):
            self.propagation(X[:-nb_class])
            self.back_propagation(X[-nb_class:])
        return 0

    def back_propagation(self, y):
        """ Update values by backpropagation, use errors """
        (output_error, hidden_errors) = self.compute_errors(y)

        for j in range(self.hidden_nb):
            for k in range(self.output_nb):
                self.vi[j][k] -= (self.learning_rate * output_error[k] * self.activation_hidden[j]) + (self.momentum * self.temp_out[j][k])
                self.temp_out[j][k] = output_error[k] * self.activation_hidden[j]

        for i in range(self.input_nb):
            for j in range(self.hidden_nb):
                self.wi[i][j] -= (self.learning_rate * hidden_errors[j] * self.activation_input[i]) + (self.momentum * self.temp_in[i][j])
                self.temp_in[i][j] = hidden_errors[j] * self.activation_input[i]

    def predict(self, X):
        """ Return a column vector with predictions for X """
        return np.array([self.propagation(x) for x in X])

    def debug(self):
        """ Print class variables for debug purposes """
        print "[DEBUG] information"
        print "input_nb:", self.input_nb
        print "hidden_nb:", self.hidden_nb
        print "output_nb:", self.output_nb
        print "learning_rate:", self.learning_rate
        print "momentum:", self.momentum
        print "activation_input:", self.activation_input
        print "activation_hidden:", self.activation_hidden
        print "activation_output:", self.activation_output
        print "wi:", self.wi
        print "vi:", self.vi
