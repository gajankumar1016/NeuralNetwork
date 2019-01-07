import numpy as np

class ActivationFunctions:
    class sigmoid:
        @staticmethod
        def compute_forward(z):
            return 1 / (1 + np.exp(-z))

        @staticmethod
        def compute_backward(z):
            return ActivationFunctions.sigmoid.compute_forward(z) * (1 - ActivationFunctions.sigmoid.compute_forward(z))

    class relu:
        @staticmethod
        def compute_forward(z):
            return np.maximum(0, z)

        @staticmethod
        def compute_backward(z):
            return np.int64(z > 0)

    class tanh:
        @staticmethod
        def compute_forward(z):
            return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

        @staticmethod
        def compute_backward(z):
            return 1 - ActivationFunctions.tanh.compute_forward(z)**2



class NNLayer:
    def __init__(self, prev_dims, num_neurons, g):
        self.numneurons = num_neurons
        self.W = np.random.randn(num_neurons, prev_dims) * np.sqrt(2 / prev_dims)
        self.b = np.random.randn(num_neurons, 1)
        self.g = g.compute_forward
        self.g_backwards = g.compute_backward
        # Z will be computed during forward prop
        self.prev_activations = None
        self.Z = None

    def forward_prop(self, prev_activations):
        Z = np.dot(self.W, prev_activations) + self.b
        self.prev_activations = prev_activations
        self.Z = Z
        return self.g(Z)



