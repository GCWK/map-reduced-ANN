import sys
import random
import classes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

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

    #plt.matshow(normalize(confusion_matrix.astype(np.float), norm='l1', axis=0))
    #plt.colorbar()
    #plt.show()

if __name__ == "__main__":
    data_import = classes.ImportData()

    if sys.argv[1] == str(0):
        np.random.seed(seed = 1)

        iris_data_x, iris_data_y = data_import.irisImport("data/iris.data", 1.0)

        iterations = 500
        input_nb = 4
        hidden_nb = 6
        learning_rate = 1.0
        output_nb = 3

        print "iterations", iterations
        print "input_nb:", input_nb
        print "hidden_nb:", hidden_nb
        print "output_nb:", output_nb
        print "learning_rate:", learning_rate

        confusion_matrices = []
        NN = classes.NeuralNetwork(input_nb, hidden_nb, num_output=output_nb, momentum=0.5, num_iter=iterations, learning_rate=learning_rate)
        NN.train(iris_data_x, iris_data_y)
        predictions = np.array([np.argmax(NN.predict(np.array([x]))) for x in iris_data_x]).reshape((iris_data_x.shape[0], 1))
        confusion_matrices.append(getConfusionMatrix(data_import.change_y_shape(iris_data_y), predictions, 3))
        evaluatePerformance(sum(confusion_matrices), 3)
    if sys.argv[1] == str(1):
        np.random.seed(seed = 1)

        breast_data_x, breast_data_y = data_import.breastCancerImport("data/breast-cancer-wisconsin.data", 1.0)

        iterations = 500
        input_nb = 9
        hidden_nb = 9
        learning_rate = 1.0
        output_nb = 2

        print "iterations", iterations
        print "input_nb:", input_nb
        print "hidden_nb:", hidden_nb
        print "output_nb:", output_nb
        print "learning_rate:", learning_rate

        confusion_matrices = []
        NN = classes.NeuralNetwork(input_nb, hidden_nb, num_output=output_nb, momentum=0.5, num_iter=iterations, learning_rate=learning_rate)
        NN.train(breast_data_x, breast_data_y)
        predictions = np.array([np.argmax(NN.predict(np.array([x]))) for x in breast_data_x]).reshape((breast_data_x.shape[0], 1))
        confusion_matrices.append(getConfusionMatrix(data_import.change_y_shape(breast_data_y), predictions, 2))
        evaluatePerformance(sum(confusion_matrices), 2)
    else:
        return "Error in parameters (0 or 1)"


    # confusion_matrices = []
    # print "input_nb:", input_nb
    # print "hidden_nb:", hidden_nb
    # print "output_nb:", output_nb
    # print "learning_rate:", learning_rate
    # for fold in kFold(iris_data_x.shape[0], 5):
    #    NN = classes.NeuralNetwork(input_nb, hidden_nb, num_output=output_nb, momentum=0.3, num_iter=50, learning_rate=learning_rate)
    #    NN.train(iris_data_x[fold[0]], iris_data_y[fold[0]])
    #    predictions = np.array([np.argmax(NN.predict(np.array([x]))) for x in iris_data_x[fold[1]]]).reshape((len(fold[1]), 1))
    #    confusion_matrices.append(getConfusionMatrix(data_import.change_y_shape(iris_data_y[fold[1]]), predictions, 3))
    # evaluatePerformance(sum(confusion_matrices), 3)

    # data_import = ImportData()
    # digits_data_x, digits_data_y = data_import.digitsImport("data/semeion.data")
    #
    # input_nb = 256
    # hidden_nb = 512
    # learning_rate = 1
    # output_nb = 10
    #
    # confusion_matrices = []
    # print "input_nb:", input_nb
    # print "hidden_nb:", hidden_nb
    # print "output_nb:", output_nb
    # print "learning_rate:", learning_rate
    # for fold in kFold(digits_data_x.shape[0], 3):
    #     NN = NeuralNetwork(input_nb, hidden_nb, num_output=output_nb, momentum=1.0, num_iter=1000, learning_rate=learning_rate)
    #     #.train(digits_data_x[fold[0]], digits_data_y[fold[0]])
    #     predictions = np.array([np.argmax(NN.predict(np.array([x]))) for x in digits_data_x[fold[1]]]).reshape((len(fold[1]), 1))
    #     confusion_matrices.append(getConfusionMatrix(data_import.change_y_shape(digits_data_y[fold[1]]), predictions, 10))
    # evaluatePerformance(sum(confusion_matrices), 10)
