import numpy as np

class ActivationFunctions:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_backwards(z):
        return ActivationFunctions.sigmoid(z) * (1 - ActivationFunctions.sigmoid(z))

    @staticmethod
    def tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_backwards(z):
        return np.int64(z > 0)

    backward_functs_dict = {"sigmoid": sigmoid_backwards, "relu":relu_backwards}
    @staticmethod
    def get_backwards_activation_function(activation_func):
        return ActivationFunctions.backward_functs_dict[activation_func]