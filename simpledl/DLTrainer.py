"""
(c) @thomaswilley, 2018
DLTrainer.py: the L-layer NN trainer for models managed by ModelManager instances

Before using, it's important to create/overload the load_data function.
It is suggested to do so by subclassing DLTrainer.

# Simple text classification DL trainer which incorporates a tokenizer (you can
# see this is reflected in the presence of 'vectorizer' attribute by default in
# ModelManager)

"""

import numpy as np
import copy

class DLTrainer(object):
    """Deep Learning / ML trainer for Models"""
    def __init__(self):
        pass

    def load_data(self, path, test_size):
        """overload this!"""
        pass

    def safelog(self, x, min_log=1e-10):
        """used just like np.log but clipped to avoid <=0 errors"""
        return np.log(x.clip(min_log))

    def nonlin_relu(self, x, deriv=False):
        """compute sigmoid or its derivative"""
        if deriv:
            return 1. * (x > 0)
        return np.maximum(0, x)

    def nonlin_sigmoid(self, z, deriv=False):
        """compute sigmoid or its derivative"""
        if deriv:
            return z * (1. - z)
        return 1. / (1. + np.exp(-1. * z))

    def cross_entropy_cost(self, Y_hat, Y):
        """compute cross-entropy cost for model output (i.e., transmute output to relative probabilities)"""
        m = Y.shape[1]
        return -1./m * np.sum(Y*self.safelog(Y_hat) + (1-Y)*self.safelog(1-Y_hat))

    def cross_entropy_cost_with_regularization(self, model, cache, Y):
        """compute overall cost including regularization"""
        l_output = len(model['shape']) - 1
        Y_hat = cache['A%d' % l_output]
        m = Y.shape[1]
        cost = self.cross_entropy_cost(Y_hat, Y)
        for i in reversed(range(1, len(model['shape']))):
            if i > 1:
                cost += 1./m * (model['lambda']/2) * (np.sum(np.square(model['w%d' % (i-1)])) \
                                                 + np.sum(np.square(model['w%d' % i])))
        return cost

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis = 0)

    def predict(self, model, X):
        """predict output of trained model on X (X needs to be (n, m))"""
        cache = self.forward(model, X)
        l_output = len(model['shape']) - 1
        Y_hat = cache['A%d' % l_output]
        dim_output = Y_hat.shape[0]
        if dim_output == 1:
            return 1. * (Y_hat >= 0.5)
        return np.argmax(self.softmax(Y_hat), axis=0)

    def correct(self, model, X, Y):
        """return percentage (/100) of correct predictions of X (as compared to Y)"""
        dim_output = Y.shape[0]
        m = X.shape[1]
        if dim_output == 1:
            pct_correct = 1./m * np.sum(1. * (self.predict(model, X) == Y))
        else:
            pct_correct = 1./m * np.sum(1. * (self.predict(model, X) == np.argmax(Y, axis=0)))
        return pct_correct

    def forward(self, model, X):
        """forward prop"""
        cache = {}
        cache['A0'] = X
        for i in range(1, len(model['shape'])):
            cache['Z%d' % i] = np.dot(model['w%d' % i], cache['A%d' % (i-1)]) + model['b%d' % i]
            cache['A%d' % i] = model['activation%d' % i](None, cache['Z%d' % i])
        return cache

    def backward(self, model, cache, X, Y):
        """backprop"""
        m = X.shape[0]
        l_output = len(model['shape']) - 1
        Y_hat = cache['A%d' % l_output]
        cache['dZ%d' % l_output] = Y_hat - Y
        cache['J'] = self.cross_entropy_cost_with_regularization(model, cache, Y)

        for i in reversed(range(1, len(model['shape']))):
            if i < l_output: # already calculated error @ output above..
                cache['dZ%d' % i] = np.dot(model['w%d' % (i+1)].T,
                                           cache['dZ%d' % (i+1)]) * model['activation%d' % i](None, cache['A%d' % i],
                                                                                              deriv=True)
            cache['dw%d' % i] = 1./m * np.dot(cache['dZ%d' % i],
                                              cache['A%d' % (i-1)].T) + (1./m * model['lambda'] * model['w%d' % i])
            cache['db%d' % i] = 1./m * np.sum(cache['dZ%d' % i], axis=1, keepdims=True)
        return cache

    def update_parameters(self, model, cache):
        """update weights and biases throughout the network"""
        for i in reversed(range(1, len(model['shape']))):
            model['w%d' % i] -= model['alpha'] * cache['dw%d' % i]
            model['b%d' % i] -= model['alpha'] * cache['db%d' % i]
        return model

    def train(self, modelmanager, X, Y, epochs, print_cost_every=0):
        """
        train the network and optionally print cost (every print_cost_every epochs; (<=0 to suppress))
        disable regularization by setting reg_lambda to 0
        """
        model = copy.deepcopy(modelmanager.model) # work on & return a copy, update modelmgr outside of trainer
        self.costs = []
        for e in range(epochs):
            cache = self.forward(model, X)
            cache = self.backward(model, cache, X, Y)

            if print_cost_every > 0:
                if e % print_cost_every == 0:
                    accuracy = self.correct(model, X, Y)
                    print("Cost after {} epochs: {} (training set accuracy: {})".format(e, cache['J'], accuracy))

            model = self.update_parameters(model, cache)

            self.costs.append(cache['J'])

        self.accuracy = self.correct(model, X, Y)

        return model, self.costs, self.accuracy

    def get_training_stats(self, model, X_dev, Y_dev):
        """return some simple stats"""
        return "dev accuracy: {:.02f}%".format(self.correct(model, X_dev, Y_dev))
