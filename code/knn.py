"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y

    def predict(self, Xtest):
        # prediction array of size Xtest
        T, D = Xtest.shape
        y_pred = np.zeros(T)
        # T x N array of euclidean distances
        X_dists = utils.euclidean_dist_squared(self.X, Xtest)

        for i, x_i in enumerate(Xtest):
            x_i_dists = X_dists[:, i]                       
            nearest_nbrs = np.argsort(x_i_dists)            
            nearest_nbrs = nearest_nbrs[:self.k]            
            y_pred[i] = utils.mode(self.y[nearest_nbrs])
        
        return y_pred
