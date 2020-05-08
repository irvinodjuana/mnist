import numpy as np
from tqdm import tqdm
import pickle


def convolution(image, filters, bias, s=1):
    """
    Apply a convolution filter(s) to an image
    """
    # dimensions of image and filter 
    # (last two dims should be same for square image/filter)
    n_f, n_c_f, f, _ = filters.shape
    n_c, in_dim, _ = image.shape
    
    out_dim = (in_dim - f) // s + 1    # calculate output dimension
    assert n_c_f == n_c     # check number of channels match

    out = np.zeros((n_f, out_dim, out_dim))

    # convolve each filter over slices of image
    for curr_f in range(n_f):
        filter = filters[curr_f]
        i_in = i_out = 0
        while i_in + f <= in_dim:
            j_in = j_out = 0
            while j_in + f <= in_dim:
                # take filter-sized slice of image, dot product with filter
                im_slice = image[:, i_in:i_in+f, j_in:j_in+f]
                out[curr_f, i_out, j_out] = np.sum(filter * im_slice) + bias[curr_f]
                j_in += s
                j_out += 1
            i_in += s
            i_out += 1

    return out


def convolution_backward(dconv_prev, conv_in, filters, s):
    """
    Compute backpropagation through convolution layer
    """
    n_f, n_c_f, f, _ = filters.shape
    _, in_dim, _ = conv_in.shape
    d_out = np.zeros(conv_in.shape)
    d_filt = np.zeros(filters.shape)
    d_bias = np.zeros((n_f, 1))

    for curr_f in range(n_f):
        i_in = i_out = 0
        while i_in + f <= in_dim:
            j_in = j_out = 0
            while j_in + f <= in_dim:
                # loss gradient of filter
                d_filt[curr_f] += dconv_prev[curr_f, i_out, j_out] * conv_in[:, i_in:i_in+f, j_in:j_in+f]
                # loss gradient of input
                d_out[:, i_in:i_in+f, j_in:j_in+f] += dconv_prev[curr_f, i_out, j_out] * filters[curr_f]
                j_in += s
                j_out += 1
            i_in += s
            i_out += 1
        # loss gradient of bias
        d_bias[curr_f] = np.sum(dconv_prev[curr_f])
    
    return d_out, d_filt, d_bias


def maxpool(image, f=2, s=2):
    """
    Apply maxpooling to an image with given window size f and stride s
    """
    # find dimensions of image, output
    n_c, h_in, w_in = image.shape
    h_out = (h_in - f) // s + 1
    w_out = (w_in - f) // s + 1
    out = np.zeros((n_c, h_out, w_out))

    for c in range(n_c):
        i_in = i_out = 0
        while i_in + f <= h_in:
            j_in = j_out = 0
            while j_in + f <= w_in:
                # save max of image window to output
                im_slice = image[:, i_in:i_in+f, j_in:j_in+f]
                out[c, i_out, j_out] = np.max(im_slice)
                j_in += s
                j_out += 1
            i_in += s
            i_out += 1
    
    return out


def nan_argmax(arr):
    idx = np.nanargmax(arr)
    return np.unravel_index(idx, arr.shape)


def maxpool_backward(d_pool, orig, f, s):
    """
    Compute backpropagation through maxpool layer
    """
    n_c, in_dim, _ = orig.shape

    d_out = np.zeros(orig.shape)

    for c in range(n_c):
        i_in = i_out = 0
        while i_in + f <= in_dim:
            j_in = j_out = 0
            while j_in + f <= in_dim:
                (a, b) = nan_argmax(orig[c, i_in:i_in+f, j_in:j_in+f])
                d_out[c, i_in+a, j_in+b] = d_pool[c, i_out, j_out]
                j_in += s
                j_out += 1
            i_in += s
            i_out += 1

    return d_out


def softmax(X):
    res = np.exp(X)
    return res/np.sum(res)


def categorical_crossentropy(y_hat, y):
    return -np.sum(y * np.log(y_hat))


def init_filter(size, scale=1.0):
    std_dev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=std_dev, size=size)


def init_weights(size):
    return 0.01 * np.random.standard_normal(size=size)



class CNN:
    """
    Convolutional Neural Network
    """
    def __init__(self):
        pass

    def funObj(self, image, label, params, conv_s, pool_f, pool_s):
        [f1, f2, w3, w4, b1, b2, b3, b4] = params

        ### Forward operation ###
        conv1 = convolution(image, f1, b1, conv_s)  # Convolution layer 1
        conv1[conv1 <= 0] = 0                       # ReLU activation

        conv2 = convolution(conv1, f2, b2, conv_s)  # Convolution layer 2
        conv2[conv2 <= 0] = 0                       # ReLU activation

        pooled = maxpool(conv2, pool_f, pool_s)     # Maxpool layer

        nf2, dim2, _ = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flattened

        z = w3.dot(fc) + b3                         # Dense layer 1 (fully-connected)
        z[z <= 0] = 0                               # ReLU activation

        out = w4.dot(z) + b4                        # Dense layer 2
        probs = softmax(out)                        # Softmax activation (probabilities)

        ### Loss ###
        f = categorical_crossentropy(probs, label)

        ### Backward operation ###
        d_out = probs - label                           # gradient of Dense layer 2
        d_w4 = d_out.dot(z.T)                           # gradient of Dense layer 2 weights
        d_b4 = np.sum(d_out, axis=1).reshape(b4.shape)  # gradient of Dense layer 2 biases

        d_z = w4.T.dot(d_out)                           # gradient of Dense layer 1
        d_z[z <= 0] = 0                                 # ReLU backprop
        d_w3 = d_z.dot(fc.T)                            # gradient of Dense layer 1 weights
        d_b3 = np.sum(d_z, axis=1).reshape(b3.shape)    # gradient of Dense layer 1 biases

        d_fc = w3.T.dot(d_z)                            # gradient of fully connected layer
        d_pool = d_fc.reshape(pooled.shape)             # reshape to pooling dims

        d_conv2 = maxpool_backward(d_pool, conv2, pool_f, pool_s)   # Maxpool backprop
        d_conv2[conv2 <= 0] = 0                                   # ReLU backprop

        d_conv1, d_f2, d_b2 = convolution_backward(d_conv2, conv1, f2, conv_s)  # Convolution 2 backprop
        d_conv1[conv1 <= 0] = 0                                               # ReLU backprop

        d_image, d_f1, d_b1 = convolution_backward(d_conv1, image, f1, conv_s)  # Convolution 1 backprop
        
        g = [d_f1, d_f2, d_w3, d_w4, d_b1, d_b2, d_b3, d_b4]

        return g, f


    def adam_GD(self, batch, num_classes, learn_rate, dim, n_c, beta1, beta2, params, cost):
        """
        Adam optimizer
        """
        [f1, f2, w3, w4, b1, b2, b3, b4] = params
        
        X = batch[:, 0:-1]
        X = X.reshape(len(batch), n_c, dim, dim)
        Y = batch[:, -1]

        cost_ = 0
        batch_size = len(batch)

        # init gradients, momentum, RMS params
        d_f1 = np.zeros(f1.shape)
        d_f2 = np.zeros(f2.shape)
        d_w3 = np.zeros(w3.shape)
        d_w4 = np.zeros(w4.shape)
        d_b1 = np.zeros(b1.shape)
        d_b2 = np.zeros(b2.shape)
        d_b3 = np.zeros(b3.shape)
        d_b4 = np.zeros(b4.shape)

        v1 = np.zeros(f1.shape)
        v2 = np.zeros(f2.shape)
        v3 = np.zeros(w3.shape)
        v4 = np.zeros(w4.shape)
        bv1 = np.zeros(b1.shape)
        bv2 = np.zeros(b2.shape)
        bv3 = np.zeros(b3.shape)
        bv4 = np.zeros(b4.shape)

        s1 = np.zeros(f1.shape)
        s2 = np.zeros(f2.shape)
        s3 = np.zeros(w3.shape)
        s4 = np.zeros(w4.shape)
        bs1 = np.zeros(b1.shape)
        bs2 = np.zeros(b2.shape)
        bs3 = np.zeros(b3.shape)
        bs4 = np.zeros(b4.shape)
        
        for i in range(batch_size):
            x = X[i]
            y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)

            g, f = self.funObj(x, y, params, 1, 2, 2)
            [d_f1_, d_f2_, d_w3_, d_w4_, d_b1_, d_b2_, d_b3_, d_b4_] = g

            d_f1 += d_f1_
            d_f2 += d_f2_
            d_w3 += d_w3_
            d_w4 += d_w4_
            d_b1 += d_b1_
            d_b2 += d_b2_
            d_b3 += d_b3_
            d_b4 += d_b4_

            cost_ += f

        v1 = beta1*v1 + (1-beta1) * d_f1 / batch_size # momentum update
        s1 = beta2*s1 + (1-beta2) * (d_f1 / batch_size)**2 # RMSProp update
        f1 -= learn_rate * v1 / np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
        
        bv1 = beta1*bv1 + (1-beta1) * d_b1 / batch_size
        bs1 = beta2*bs1 + (1-beta2) * (d_b1 / batch_size)**2
        b1 -= learn_rate * bv1 / np.sqrt(bs1+1e-7)
    
        v2 = beta1*v2 + (1-beta1) * d_f2 / batch_size
        s2 = beta2*s2 + (1-beta2) * (d_f2 / batch_size)**2
        f2 -= learn_rate * v2 / np.sqrt(s2+1e-7)
                        
        bv2 = beta1*bv2 + (1-beta1) * d_b2 / batch_size
        bs2 = beta2*bs2 + (1-beta2) * (d_b2 / batch_size)**2
        b2 -= learn_rate * bv2 / np.sqrt(bs2+1e-7)
        
        v3 = beta1*v3 + (1-beta1) * d_w3 / batch_size
        s3 = beta2*s3 + (1-beta2) * (d_w3 / batch_size)**2
        w3 -= learn_rate * v3 / np.sqrt(s3+1e-7)
        
        bv3 = beta1*bv3 + (1-beta1) * d_b3 / batch_size
        bs3 = beta2*bs3 + (1-beta2) * (d_b3 / batch_size)**2
        b3 -= learn_rate * bv3 / np.sqrt(bs3+1e-7)
        
        v4 = beta1*v4 + (1-beta1) * d_w4 / batch_size
        s4 = beta2*s4 + (1-beta2) * (d_w4 / batch_size)**2
        w4 -= learn_rate * v4 / np.sqrt(s4+1e-7)
        
        bv4 = beta1*bv4 + (1-beta1) * d_b4 / batch_size
        bs4 = beta2*bs4 + (1-beta2) * (d_b4 / batch_size)**2
        b4 -= learn_rate * bv4 / np.sqrt(bs4+1e-7)

        cost_ = cost_ / batch_size
        cost.append(cost_)
        params = [f1, f2, w3, w4, b1, b2, b3, b4]

        return params, cost


    def fit(
        self, X, y, 
        num_classes=10, learn_rate=0.01, beta1=0.95, beta2=0.99, 
        img_dim=28, img_depth=1, f=5, num_filter1=8, num_filter2=8,
        batch_size=32, num_epochs=2, save_path='../data/params.pkl'
    ):
        X *= 256.

        print(np.mean(X), np.std(X))

        X -= int(np.mean(X))
        X /= int(np.std(X))

        y = np.expand_dims(y, axis=1)
    
        data = np.hstack((X, y))
        np.random.shuffle(data)

        # Init parameters
        f1 = init_filter((num_filter1 ,img_depth,f,f))
        f2 = init_filter((num_filter2, num_filter1,f,f))
        w3 = init_weights((128, 800))
        w4 = init_weights((10, 128))

        b1 = np.zeros((f1.shape[0], 1))
        b2 = np.zeros((f2.shape[0], 1))
        b3 = np.zeros((w3.shape[0], 1))
        b4 = np.zeros((w4.shape[0], 1))

        params = [f1, f2, w3, w4, b1, b2, b3, b4]
        cost = []

        print(f"Learn Rate: {learn_rate}, Batch Size: {batch_size}")

        for epoch in range(num_epochs):
            np.random.shuffle(data)
            batches = [data[k:k+batch_size] for k in range(0, data.shape[0], batch_size)]

            t = tqdm(batches)

            for x, batch in enumerate(t):
                params, cost = self.adam_GD(
                    batch, num_classes, learn_rate, img_dim, img_depth,
                    beta1, beta2, params, cost
                )
                t.set_description("Cost: %.3f" % (cost[-1]))

        # Save parameters (weights) in memory and on disk
        self.params = params

        with open(save_path, 'wb') as file:
            pickle.dump([params, cost], file)
        
        return cost
    
    def load_weights(self, save_path='../data/cnn_params.pkl'):
        params, cost = pickle.load(open(save_path, 'rb'))
        self.params = params

    def predict(self, X, num_classes=10, img_dim=28, conv_s=1, pool_f=2, pool_s=2):
        [f1, f2, w3, w4, b1, b2, b3, b4] = self.params
        n, d = X.shape
        
        X *= 256.
        X -= int(np.mean(X))
        X /= int(np.std(X))
        
        X = X.reshape(n, 1, img_dim, img_dim)
        y_hat = np.zeros((n, num_classes))

        for i in range(n):
            x = X[i]
            conv1 = convolution(x, f1, b1, conv_s)  # Convolution layer 1
            conv1[conv1 <= 0] = 0                       # ReLU activation

            conv2 = convolution(conv1, f2, b2, conv_s)  # Convolution layer 2
            conv2[conv2 <= 0] = 0                       # ReLU activation

            pooled = maxpool(conv2, pool_f, pool_s)     # Maxpool layer

            nf2, dim2, _ = pooled.shape
            fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flattened

            z = w3.dot(fc) + b3                         # Dense layer 1 (fully-connected)
            z[z <= 0] = 0                               # ReLU activation

            out = w4.dot(z) + b4                        # Dense layer 2
            probs = softmax(out)                        # Softmax activation (probabilities)
            y_hat[i] = out.flatten()

            print(f"Predicting {i} of {n} examples")

        return np.argmax(y_hat, axis=1)



    