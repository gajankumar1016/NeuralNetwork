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





