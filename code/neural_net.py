import numpy as np
import findMin
import utils

# helper functions to transform between one big vector of weights
# and a list of layer parameters of the form (W,b) 
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights,())])

def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes)-1):
        W_size = layer_sizes[i+1] * layer_sizes[i]
        b_size = layer_sizes[i+1]
        
        W = np.reshape(weights_flat[counter:counter+W_size], (layer_sizes[i+1], layer_sizes[i]))
        counter += W_size

        b = weights_flat[counter:counter+b_size][None]
        counter += b_size

        weights.append((W,b))  
    return weights

def log_sum_exp(Z):
    Z_max = np.max(Z,axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:,None]), axis=1)) # per-colmumn max

class NeuralNet():
    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes, lammy=1, max_iter=100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy
        self.max_iter = max_iter

    def funObj(self, weights_flat, X, y):
        weights = unflatten_weights(weights_flat, self.layer_sizes)

        activations = [X]
        for W, b in weights:
            Z = X @ W.T + b
            X = 1/(1+np.exp(-Z))
            activations.append(X)

        yhat = Z
        
        if self.classification: # softmax- TODO: use logsumexp trick to avoid overflow
            tmp = np.sum(np.exp(yhat), axis=1)
            # f = -np.sum(yhat[y.astype(bool)] - np.log(tmp))
            f = -np.sum(yhat[y.astype(bool)] - log_sum_exp(yhat))
            grad = np.exp(yhat) / tmp[:,None] - y
        else:  # L2 loss
            f = 0.5*np.sum((yhat-y)**2)  
            grad = yhat-y # gradient for L2 loss

        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)

        g = [(grad_W, grad_b)]

        for i in range(len(self.layer_sizes)-2,0,-1):
            W, b = weights[i]
            grad = grad @ W
            grad = grad * (activations[i] * (1-activations[i])) # gradient of logistic loss
            grad_W = grad.T @ activations[i-1]
            grad_b = np.sum(grad,axis=0)

            g = [(grad_W, grad_b)] + g # insert to start of list

        g = flatten_weights(g)
        
        # add L2 regularization
        f += 0.5 * self.lammy * np.sum(weights_flat**2)
        g += self.lammy * weights_flat 
        
        return f, g

    
    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:,None]
            
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1]>1 # assume it's classification iff y has more than 1 column

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes)-1):
            W = scale * np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i])
            b = scale * np.random.randn(1,self.layer_sizes[i+1])
            weights.append((W,b))
        weights_flat = flatten_weights(weights)

        # utils.check_gradient(self, X, y, len(weights_flat), epsilon=1e-6)
        weights_flat_new, f = findMin.findMin(self.funObj, weights_flat, self.max_iter, X, y, verbose=True)

        self.weights = unflatten_weights(weights_flat_new, self.layer_sizes)


        # self.weights = unflatten_weights(weights_flat, self.layer_sizes)
        # print("X", X.shape)
        # print("y", y.shape)
        # for i, w in enumerate(weights):
        #     print("W" + str(i), w[0].T.shape)

    def fit_SGD(self, X, y):
        NUM_EPOCHS = 10
        MINIBATCH_SIZE = 500
        ALPHA = 0.001

        # copied from fit (up to create minibatches)
        if y.ndim == 1:
            y = y[:,None]
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1]>1 # assume it's classification iff y has more than 1 column
        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes)-1):
            W = scale * np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i])
            b = scale * np.random.randn(1,self.layer_sizes[i+1])
            weights.append((W,b))
        weights_flat = flatten_weights(weights)

        # create minibatches
        n, d = X.shape
        n, k = y.shape
        data = np.hstack((X, y))    # X and y horizontally concat'd to shuffle together
        np.random.shuffle(data)
        n_batches = n // MINIBATCH_SIZE
        minibatches = np.split(data, n_batches)

        c = 0

        # stochastic gradient descent
        for i in range(NUM_EPOCHS):
            for batch in minibatches:
                X_batch = batch[:, 0:d]     # separate X and y from batch
                y_batch = batch[:, d:d+k]
                f, g = self.funObj(weights_flat, X_batch, y_batch)
                weights_flat = weights_flat - ALPHA * g
                print(c, "loss: ", f)
                c += 1
        
        self.weights = unflatten_weights(weights_flat, self.layer_sizes)


    def predict(self, X):
        for W, b in self.weights:
            Z = X @ W.T + b
            X = 1/(1+np.exp(-Z))
        if self.classification:
            return np.argmax(Z,axis=1)
        else:
            return Z




