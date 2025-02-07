
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import math


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer) # only one hidden layer
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        pass   #TODO: implement

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return x * (1 - x)
        pass   #TODO: implement

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        x_batch = xVals
        y_batch = yVals
        for i in range(0, epochs):
            if minibatches:
                x_batch = self.__batchGenerator(xVals, mbs)
                y_batch = self.__batchGenerator(yVals, mbs)
            
            for x_cur, y_cur in zip(x_batch, y_batch):
                # forward pass
                l1, l2 = self.__forward(x_cur)

                #start of backprop
                sig_prime = [self.__sigmoidDerivative(l1), self.__sigmoidDerivative(l2)]
                activation_cost = l2 - y_cur
                l2d = activation_cost * sig_prime[1]
                l1e = np.dot(l2d , self.W2.transpose())
                l1d = l1e * sig_prime[0]
                
                self.W1 = self.W1 - ((self.lr / mbs) * np.dot(x_cur.transpose(), l1d))
                self.W2 = self.W2 - ((self.lr / mbs) * np.dot(l1.transpose(), l2d))

        pass          #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    temp = []
    for i in xTrain:
        temp.append(np.ndarray.flatten(i))
    xTrain = np.array(temp) / 255
    temp = []
    for i in xTest:
        temp.append(np.ndarray.flatten(i))
    xTest = np.array(temp) / 255

    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))

# https://www.tensorflow.org/tutorials/keras/classification

def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        model = NeuralNetwork_2Layer(784, 10, 384, 10)
        model.train(xTrain, yTrain, 10)
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        model = keras.Sequential([
            keras.layers.Dense(512, activation='sigmoid'),
            keras.layers.Dense(10)
        ])
        model.compile(optimizer='adam', 
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=10)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def normalize(data):
    for i in range(len(data)): 
        temp = np.argmax(data[i])
        for j in range(len(data[i])):
            if j == temp:
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data

def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        return normalize(model.predict(data))
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        return normalize(model.predict(data))
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    conf = {}
    fp = {}
    fn = {}
    for i in range(0, 10):
        conf[str(i)] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fp[str(i)] = 0
        fn[str(i)] = 0
    for i in range(preds.shape[0]):
        conf[str(np.argmax(yTest[i]))][np.argmax(preds[i])] += 1
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    print('o\\p{:5d} {:5d} {:5d} {:5d} {:5d} {:5d} {:5d} {:5d} {:5d} {:5d}'.format(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    for i in range(0, 10):
        temp = conf[str(i)]
        # true positives + false negatives
        fn[str(i)] = sum(temp)

        # get true positives + false positives
        t2 = []
        for j in range(0, 10):
            t2.append(conf[str(j)][i])
        fp[str(i)] = sum(t2)
        print('{}: {:5d} {:5d} {:5d} {:5d} {:5d} {:5d} {:5d} {:5d} {:5d} {:5d}'.format(i, temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7], temp[8], temp[9]))
    
    print()
    print('    precision\t\trecall\t\tf1 score')
    for i in range(0, 10):
        try:
            prec = conf[str(i)][i] / fp[str(i)]
        except:
            prec = 0

        try:
            rec = conf[str(i)][i] / fn[str(i)]
        except:
            rec = 0

        try:
            f1 = (2 * prec * rec)/(prec + rec)
        except:
            f1 = 0
        print('{}:\t{:01.3f}\t\t {:2.3f}\t\t   {:3.3f}'.format(i, prec, rec, f1))



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
