import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)


class logRegL2(logReg):
    def __init__(self, verbose=0, lammy=1.0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.bias = True

    # L2 Regularized Logistic Regression
    def funObj(self, w, X, y):
        yXw = y * (X@w)
        reg = (self.lammy / 2.) * np.sum(w ** 2)
        
        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))
        f = f + reg

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)
        g = g + (self.lammy * w)

        return f, g
    
    def fit_SGD(self, X, y):
        NUM_EPOCHS = 10
        MINIBATCH_SIZE = 500
        ALPHA = 0.001

        # create minibatches
        y = y.reshape((-1, 1))      # convert y to 2D to make it stackable
        n, d = X.shape
        n, k = y.shape
        data = np.hstack((X, y))    # X and y horizontally concat'd to shuffle together
        np.random.shuffle(data)
        n_batches = n // MINIBATCH_SIZE
        minibatches = np.split(data, n_batches)

        # Initial guess
        self.w = np.zeros(d)
        iter = 0

        # stochastic gradient descent
        for i in range(NUM_EPOCHS):
            for batch in minibatches:
                # decreasing step size
                iter += 1
                # ALPHA = 0.001 / np.sqrt(iter)
                # ALPHA = 0.001

                X_batch = batch[:, 0:d]             # separate X and y from batch
                y_batch = batch[:, -1].flatten()    # convert y back to 1D
                f, g = self.funObj(self.w, X_batch, y_batch)
                self.w = self.w - ALPHA * g
                
                if self.verbose and iter % 100 == 0:
                    print(iter, "loss: ", f)


class logRegL1(logReg):
    # L1 Regularized Logistic Regression
    def __init__(self, verbose=0, L1_lambda=1.0, maxEvals=100):
        self.verbose = verbose
        self.L1_lambda = L1_lambda
        self.maxEvals = maxEvals
        self.bias = True

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda,
                                        self.maxEvals, X, y, verbose=self.verbose)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the seected set

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature
                XSelect = X[:, list(selected_new)]
                model = logReg()
                model.fit(XSelect, y)
                yXw = y * (XSelect@model.w)
                
                loss = np.sum(np.log(1. + np.exp(-yXw)))
                loss += self.L0_lambda * np.sum(model.w != 0)

                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


def kernel_RBF(X1, X2, sigma=1):
    D = utils.euclidean_dist_squared(X1, X2)
    return np.exp(-D / (2 * sigma**2))

def kernel_none(X1, X2, **kernel_args):
    return X1


class logLinearClassifier:
    def __init__(self, lammy=1.0, maxEvals=100, verbose=True, kernel_fun=kernel_RBF, **kernel_args):
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.verbose = verbose
        self.kernel_fun = kernel_fun
        self.kernel_args = kernel_args

    def fit(self, X, y):
        # n = num examples, d = num features, k = num classes
        self.X = X
        K = self.kernel_fun(X, self.X, **self.kernel_args)

        n, d = K.shape
        k = np.unique(y).size

        # Initial guess
        self.W = np.zeros((k, d))

        for i in range(k):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            log_model = logRegL2(lammy=self.lammy, maxEvals=self.maxEvals, verbose=self.verbose)
            log_model.fit_SGD(K, ytmp)
            self.W[i] = log_model.w
        
    def predict(self, X):
        K = self.kernel_fun(X, self.X, **self.kernel_args)
        return np.argmax(K@self.W.T, axis=1)


class softmaxClassifier:
    def __init__(self, maxEvals, verbose=False):
        self.maxEvals = maxEvals
        self.verbose = verbose
    
    def funObj(self, w, X, y):
        n, d = X.shape
        k = int(np.size(w) / d)
        W = w.reshape((k, d))

        # Calculate function value
        f = 0
        for i in range(n):
            f += -W[y[i]].T@X[i]
            f += np.log(np.sum(np.exp(X[i]@W.T)))
        
        # Calculate gradient value
        XWT = X@W.T
        exp_XWT = np.exp(XWT)
        denoms = np.sum(exp_XWT, axis=1)

        g = np.zeros((k, d))
        for c in range(k):
            for j in range(d):
                g[c, j] = np.sum(-X[:, j]*(y==c) + X[:, j]*exp_XWT[:, c] / denoms)

        g = g.flatten()
        return f, g
    
    def fit(self, X, y):
        n, d = X.shape
        k = np.unique(y).size

        # Initial guess, flattened
        self.w = np.zeros(k*d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        n, d = X.shape
        k = int(np.size(self.w) / d)
        W = self.w.reshape((k, d))
        return np.argmax(X@W.T, axis=1)

