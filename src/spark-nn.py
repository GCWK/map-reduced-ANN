import classes
import numpy as np
from pyspark import SparkContext
import sys

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
    total = np.sum(confusion_matrix)
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


if __name__ == "__main__":
    # Initialize Spark Instance
    sc = SparkContext(appName = "BPNN")
    np.random.seed(seed = 1)
    workers_nb = 4

    # Import classes and data
    data_import = classes.ImportData()
    iris_data_x, iris_data_y = data_import.irisImport(sys.argv[1], 1.0)

    # Set parameters
    iterations = 3000
    input_nb = 4
    hidden_nb = 6
    learning_rate = 1.0
    output_nb = 3

    NN = sc.broadcast(classes.NeuralNetwork(input_nb, hidden_nb, num_output=output_nb, momentum=0.5, num_iter=iterations, learning_rate=learning_rate))
    test_data = sc.broadcast(iris_data_x)

    iris_data_xy = np.random.permutation(np.concatenate((iris_data_x, iris_data_y), axis=1))
    data_x, data_y = sc.parallelize(iris_data_x, workers_nb), sc.parallelize(iris_data_y, workers_nb)
    data_xy = sc.parallelize(iris_data_xy, workers_nb)

    data_train = data_xy.map(lambda x: NN.value.train_unique(x))
    print "Count:", data_train.count()

    data_test1 = sc.parallelize(iris_data_x, 1).zipWithIndex().map(lambda (k, v): (v, k))
    data_test2 = sc.parallelize(iris_data_x, 1).zipWithIndex().map(lambda (k, v): (v, k))
    data_test3 = sc.parallelize(iris_data_x, 1).zipWithIndex().map(lambda (k, v): (v, k))
    data_test4 = sc.parallelize(iris_data_x, 1).zipWithIndex().map(lambda (k, v): (v, k))
    data_predicted1 = data_test1.map(lambda x: np.argmax(NN.value.predict(np.array([x[1]])))).zipWithIndex()
    data_predicted2 = data_test2.map(lambda x: np.argmax(NN.value.predict(np.array([x[1]])))).zipWithIndex()
    data_predicted3 = data_test3.map(lambda x: np.argmax(NN.value.predict(np.array([x[1]])))).zipWithIndex()
    data_predicted4 = data_test4.map(lambda x: np.argmax(NN.value.predict(np.array([x[1]])))).zipWithIndex()

    nn_vote = data_predicted1.union(data_predicted2.union(data_predicted3.union(data_predicted4))) \
                          .map(lambda (k, v): (v, k)) \
                          .mapValues(lambda x: {x}) \
                          .reduceByKey(lambda s1, s2: np.bincount(np.array(np.append(list(s1),list(s2)))))

    for i in nn_vote.collect():
        print i

    for i in nn_vote.mapValues(lambda x: x.argmax()).sortByKey().collect():
        print i

    results = np.array([x[1] for x in nn_vote.mapValues(lambda x: x.argmax()).sortByKey().collect()]).reshape((iris_data_x.shape[0], 1))

    confusion_matrices = []
    confusion_matrices.append(getConfusionMatrix(data_import.change_y_shape(iris_data_y), results, 3))
    evaluatePerformance(sum(confusion_matrices), 3)
