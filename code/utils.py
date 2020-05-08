import os
import sys
import time
import os.path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
from scipy.optimize import approx_fprime


def plotClassifier(model, X, y, fname):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line =  np.linspace(x1_min, x1_max, 200)
    x2_line =  np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    y_pred = np.reshape(y_pred, x1_mesh.shape)

    plt.figure()
    plt.xlim([x1_mesh.min(), x1_mesh.max()])
    plt.ylim([x2_mesh.min(), x2_mesh.max()])

    plt.contourf(x1_mesh, x2_mesh, -y_pred.astype(int), # unsigned int causes problems with negative sign... o_O
                cmap=plt.cm.RdBu, alpha=0.6)

    plt.scatter(x1[y==0], x2[y==0], color="b", label="class 0")
    plt.scatter(x1[y==1], x2[y==1], color="r", label="class 1")
    plt.legend()

    fname = os.path.join("..", "figs", fname)
    plt.savefig(fname)
    print("Saved plot to: " + fname)


def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y)==0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)

    # without broadcasting:
    # n,d = X.shape
    # t,d = Xtest.shape
    # D = X**2@np.ones((d,t)) + np.ones((n,d))@(Xtest.T)**2 - 2*X@Xtest.T


def check_gradient(model, X, y, dimensionality=None, verbose=True):
    # This checks that the gradient implementation is correct
    if not dimensionality:
        dimensionality = model.w.size
    w = np.random.rand(dimensionality)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-3):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        if verbose:
            print('User and numerical derivatives agree.')


def classification_error(y, yhat):
    return np.mean(y!=yhat)


def random_select(X, y, n):
    # Randomly selects n samples of X and y, paired by index
    assert len(X) == len(y)
    p = np.random.permutation(len(X))[:n]
    return X[p], y[p]


def one_hot_encoding(y):
    # Creates a one-hot encoding for 1D array y
    n = y.size
    k = y.max() + 1
    Y = np.zeros((n, k))
    rows = np.arange(n)
    Y[rows, y] = 1
    return Y


def validation_summary(name, model, X, y, k=5):
    # Performs k-fold cross-validation for models (with random shuffling)
    n, d = X.shape
    assert n % k == 0
    assert n == len(y)

    valid_errs = np.zeros(k)
    p = np.random.permutation(len(X))
    batch_size = n // k

    for i in range(k):
        start, end = i * batch_size, (i+1) * batch_size
        train_index = np.append(p[:start], p[end:])
        valid_index = p[start:end]

        X_train, y_train = X[train_index], y[train_index]
        X_valid, y_valid = X[valid_index], y[valid_index]

        model.fit(X_train, y_train)
        y_hat = model.predict(X_valid)
        valid_errs[i] = classification_error(y_hat, y_valid)
        print(f"{name} validation error {i+1} of {k}: {valid_errs[i]:.3f}")

    err = np.mean(valid_errs)
    print(f"{name} Cross Validation Error: {err:.3f}")
    return err


def test_summary(name, model, X, y, Xtest, ytest, train=True):
    # Final evaluation train/test errors and times
    start = time.time()

    if train:
        model.fit(X, y)

    train_time = time.time() - start

    y_pred_tr = model.predict(X)
    y_pred_te = model.predict(Xtest)
    train_err = classification_error(y_pred_tr, y)
    test_err = classification_error(y_pred_te, ytest)

    if start:
        predict_time = time.time() - start - train_time
    print()
    print(f"{name} train error: {train_err:.3f}")
    print(f"{name} test error: {test_err:.3f}")

    if start:
        if train_time < 60:
            print(f"Training time: {train_time:.3f} sec")
        else:
            print(f"Training time: {(train_time/60):.3f} min")
        if predict_time < 60:
            print(f"Prediction time: {predict_time:.3f} sec")
        else:
            print(f"Prediction time: {(predict_time/60):.3f} min")
    