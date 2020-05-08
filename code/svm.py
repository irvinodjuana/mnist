import numpy as np
import utils


def kernel_RBF(X1, X2, sigma=1):
    D = utils.euclidean_dist_squared(X1, X2)
    return np.exp(-D / (2 * sigma**2))

def kernel_poly(X1, X2, p=2):
    return np.power(X1@X2.T + 1, p)

def kernel_none(X1, X2, **kernel_args):
    return X1


class BinarySVM:
    """ Binary Support Vector Machine Model """
    def __init__(self, lammy=1.0, verbose=0):
        self.lammy = lammy
        self.verbose = verbose
    
    def funObj(self, w, X, y):
        n, d = X.shape
        yXw = y * (X@w)
        zero = np.zeros(n)

        # hinge loss values
        hinge = np.maximum(zero, 1-yXw)

        # calculate function value: hinge loss + L2 reg
        f = np.sum(hinge) + (self.lammy / 2.) * np.sum(w ** 2)

        # calculate gradient
        y_partial = -y
        y_partial[hinge==0.] = 0.
        g = X.T@y_partial + self.lammy * w

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
                    

    def predict(self, X):
        return np.sign(X@self.w)


class SVM:
    """ Multiclass SVM Model """
    def __init__(self, lammy=1.0, verbose=0, kernel_fun=kernel_RBF, **kernel_args):
        self.lammy = lammy
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
            if self.verbose:
                print("\nTraining class %d" % i)
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            binary_model = BinarySVM(lammy=self.lammy, verbose=self.verbose)
            binary_model.fit_SGD(K, ytmp)
            self.W[i] = binary_model.w
    
    def predict(self, X):
        K = self.kernel_fun(X, self.X, **self.kernel_args)
        return np.argmax(K@self.W.T, axis=1)