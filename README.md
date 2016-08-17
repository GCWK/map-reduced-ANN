# map-reduced-ANN
Implementation of an artificial neural network parallelized in Spark

# How to launch the code:
Spark must be installed on the computer executing the code.

1. Download Spark here: http://spark.apache.org/downloads.html
2. In the root of Spark folder, execute `build/mvn -Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -DskipTests clean package` to build it.
3. Add an alias to ./bin/spark-submit for spark-submit.

Then, in the root of the python project:

  To test the parallelized neural network on Iris Dataset:
  `spark-submit --py-files ./src/classes.py ./src/spark-nn.py ./data/iris.data 0`

  To test the parallelized neural network on Breast Cancer Dataset:
  `spark-submit --py-files ./src/classes.py ./src/spark-nn.py ./data/iris.data 1`

In order to compare, the classical neural network can be executing:
  `python src/neural-network.py 0` for Iris Dataset
  `python src/neural-network.py 1` for Breast Cancer Dataset
