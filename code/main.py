import os
import pickle
import gzip
import argparse
import numpy as np
import time

import utils

import knn
import linear_model
import svm
import neural_net
import cnn


# Helper methods 

def load_mnist():
    with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    X, y = train_set
    Xtest, ytest = test_set
    return X, y, Xtest, ytest

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

# Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', required=True)

    io_args = parser.parse_args()
    model = io_args.model
    
    if model == "1":
        X, y, Xtest, ytest = load_mnist()

        Y = utils.one_hot_encoding(y)

        print("X, y, Y", X.shape, y.shape, Y.shape)
        print("Xtest, ytest", Xtest.shape, ytest.shape)
    
    elif model == "1.1":
        """KNN Model"""
        X, y, Xtest, ytest = load_mnist()

        model = knn.KNN(k=3)

        # utils.validation_summary("KNN", model, X, y, k=5)
        utils.test_summary("KNN", model, X, y, Xtest, ytest)
    
    elif model == "1.2":
        """Linear Model"""
        X, y, Xtest, ytest = load_mnist()

        # one vs. all logistic regression model
        model = linear_model.logLinearClassifier(
            lammy=1.0, 
            maxEvals=500, 
            verbose=1,
            kernel_fun=linear_model.kernel_RBF,
            sigma=0.8
        )

        # utils.validation_summary("Linear Model", model, X, y, k=5)
        utils.test_summary("Linear Model", model, X, y, Xtest, ytest)

    elif model == "1.3":
        """SVM Model"""
        X, y, Xtest, ytest = load_mnist()

        # svm model
        model = svm.SVM(
            lammy=0.001, 
            verbose=0, 
            kernel_fun=svm.kernel_RBF,
            sigma=0.8
            # p=5
        )

        # utils.validation_summary("SVM", model, X, y, k=5)
        utils.test_summary("SVM", model, X, y, Xtest, ytest)

    elif model == "1.4":
        """MLP Model"""
        X, y, Xtest, ytest = load_mnist()
        Y = utils.one_hot_encoding(y)

        hidden_layer_sizes = [100]
        print(hidden_layer_sizes)
        model = neural_net.NeuralNet(hidden_layer_sizes, lammy=0.1, max_iter=500)

        # utils.validation_summary("MLP", model, X, Y, k=5)
        utils.test_summary("MLP", model, X, Y, Xtest, ytest)
    

    elif model == "1.5":
        """CNN Model"""

        start = time.time()
        X, y, Xtest, ytest = load_mnist()

        model = cnn.CNN()
        model.fit(X, y, save_path='../data/cnn_params.pkl')
        # model.load_weights(save_path='../data/cnn_params.pkl')

        # utils.validation_summary("CNN", model, X, y, k=5)
        utils.test_summary("CNN", model, X, y, Xtest, ytest, train=False)
        

    else:
        print("Unknown model: %s" % model)    