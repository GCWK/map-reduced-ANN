import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

class ImportData:
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


class NeuralNetwork:
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
        self.activation_input = copy.copy(X)

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
        print "\n[DEBUG] information"
        print "input_nb:", self.input_nb
        print "hidden_nb:", self.hidden_nb
        print "output_nb:",self.output_nb
        print "learning_rate:", self.learning_rate
        print "momentum:", self.momentum
        print "activation_input:", self.activation_input
        print "activation_hidden:", self.activation_hidden
        print "activation_output:", self.activation_output
        print "wi:", self.wi
        print "vi:", self.vi


def kFold(size, folds=10):
    """ Return indices for kfolds """
    for i in xrange(folds):
        test_indexes = np.array([j for j in range(size) if j % folds == i])
        train_indexes = np.array([j for j in range(size) if j % folds != i])
        yield (train_indexes, test_indexes)

def getConfusionMatrix(y, y_hat, num_classes):
    """ take two column class vectors and print its confusion matrix """
    assert(y.shape[0] == y_hat.shape[0] and y.shape[1] == y_hat.shape[1])
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(y.shape[0]):
        confusion_matrix[y[i], y_hat[i]] += 1
    return confusion_matrix

def evaluatePerformance(confusion_matrix, num_classes):
    """ Return the confusion matrix, accuracy, precision, recall and F-measure """
    total = sum([sum(x) for x in confusion_matrix])
    accuracy = float(np.trace(confusion_matrix)) / float(total)
    precision = []
    recall = []
    f_measure = []
    for i in range(num_classes):
        precision.append(float(confusion_matrix[i, i]) / float(sum(confusion_matrix[i, :])))
        try:
            recall.append(float(confusion_matrix[i, i]) / float(sum(confusion_matrix[:, i])))
        except:
            recall.append(0.0)
        try:
            f_measure.append(float(2*precision[-1]*recall[-1])/float(precision[-1]+recall[-1]))
        except:
            f_measure.append(0.0)

    print "Confusion matrix:\n", confusion_matrix
    print "Accuracy:\n", np.around(accuracy, decimals=3)
    print "Precision:\n", np.around(precision, decimals=3)
    print "Recall:\n", np.around(recall, decimals=3)
    print "F-measure:\n", np.around(f_measure, decimals=3)
    print "\n"

    plt.matshow(normalize(confusion_matrix.astype(np.float), norm='l1', axis=0))
    plt.colorbar()
    plt.show()

## Parameters exploration
##########################

data_import = ImportData()
iris_data_x, iris_data_y = data_import.irisImport("data/iris.data", 1.0)

input_nb = 4
hidden_nb = 5
learning_rate = 1.0
output_nb = 3

confusion_matrices = []
print "input_nb:", input_nb
print "hidden_nb:", hidden_nb
print "output_nb:", output_nb
print "learning_rate:", learning_rate
for fold in kFold(iris_data_x.shape[0], 5):
    NN = NeuralNetwork(input_nb, hidden_nb, num_output=output_nb, momentum=0.3, num_iter=50, learning_rate=learning_rate)
    NN.train(iris_data_x[fold[0]], iris_data_y[fold[0]])
    predictions = np.array([np.argmax(NN.predict(np.array([x]))) for x in iris_data_x[fold[1]]]).reshape((len(fold[1]), 1))
    confusion_matrices.append(getConfusionMatrix(data_import.change_y_shape(iris_data_y[fold[1]]), predictions, 3))
evaluatePerformance(sum(confusion_matrices), 3)


#data_import = ImportData()
#digits_data_x, digits_data_y = data_import.digitsImport("data/semeion.data")

#hidden_nb = 20
#learning_rate = 1.0
#output_nb = 10

#confusion_matrices = []
#print "hidden_nb:", hidden_nb
#print "learning_rate:", learning_rate
#for fold in kFold(digits_data_x.shape[0], 3):
#    NN = NeuralNetwork(4, hidden_nb, num_output=output_nb, momentum=0.3, num_iter=100, learning_rate=learning_rate)
#    #.train(digits_data_x[fold[0]], digits_data_y[fold[0]])
#    predictions = np.array([np.argmax(NN.predict(np.array([x]))) for x in digits_data_x[fold[1]]]).reshape((len(fold[1]), 1))
#    confusion_matrices.append(getConfusionMatrix(data_import.change_y_shape(digits_data_y[fold[1]]), predictions, 10))
#evaluatePerformance(sum(confusion_matrices), 10)
