"""
(c) @thomaswilley, 2018
ModelManager.py: A simple utility to load, save, and describe models.

self.model is a simple dict which is pickled to/from disk so it can readily
include vectorizers or other transforms/functions
"""

import numpy as np
import pickle

class ModelManager(object):
    """Simple Neural Network Model Container"""

    def __init__(self, path=None):
        self.model = None
        self.path = path
        if self.path:
            self.model = self.load_model(self.path)

    def create_model(self, dims=None, activations=None, vectorizer=None, default_alpha=None, default_lambda=None):
        """initialize model - note that activations & vectorizer must be tucked inside model for pickling"""
        assert(len(activations) == len(dims)-1) # network should have L-l activations
        model = {}
        model['shape'] = dims
        model['vectorizer'] = vectorizer
        model['alpha'] = default_alpha
        model['lambda']= default_lambda
        for i in range(1, len(dims)):
            model['w%d' % i] = np.random.random((dims[i], dims[i-1]))
            model['b%d' % i] = np.zeros((dims[i], 1))
            model['activation%d' % i] = activations[i-1]

        self.model = model

    def update_model(self, model):
        """update the model object directly"""
        self.model = model

    def _generate_random_path(self):
        return 'model_{}.inpy'.format(np.random.randint(10000))

    def save_model(self, path=None, overwrite_ok=False):
        """save model to path (optionally overwriting in place)"""
        if self.path and overwrite_ok: # overwrite
            path = self.path

        if self.path and not overwrite_ok:
            path = self._generate_random_path()

        if self.path is None:
            self.path = self._generate_random_path()
            path = self.path

        with open(path, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return path

    def load_model(self, path):
        """load model from path"""
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
            for k in data.keys():
                if k[0] == 'b': # reshape bias terms
                    data[k] = data[k].reshape(-1, 1)

            self.model = data
            return data

    def get_shapes(self):
        """get a quick string of dimensions of model attributes which have the shape attribute defined"""
        return "{}".format(str([
            (k, self.model[k].shape) for k in self.model.keys() if hasattr(self.model[k], 'shape')]))
