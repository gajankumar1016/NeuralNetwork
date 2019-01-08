import numpy as np

class NNLayer:
    def __init__(self):
        pass


class Dense(NNLayer):
    def __init__(self, num_neurons, activation, initialization=None):
        super().__init__()
        self.g = activation.compute_forward
        self.g_backwards = activation.compute_backward
        self.num_neurons = num_neurons
        self.initialization = initialization
        if activation == "relu" and not initialization:
            self.initialization = "He"

        # Z will be computed during forward prop
        self.prev_activations = None
        self.Z = None

    def _initialize(self, prev_dims):
        self.W = np.random.randn(self.num_neurons, prev_dims)
        if self.initialization == "He":
            self.W *= np.sqrt(2 / prev_dims)
        elif self.initialization == "xavier":
            self.W *= np.sqrt(1 / prev_dims)

        self.b = np.random.randn(self.num_neurons, 1)

    def forward_prop(self, prev_activations):
        Z = np.dot(self.W, prev_activations) + self.b
        self.prev_activations = prev_activations
        self.Z = Z
        return self.g(Z)


class Flatten(NNLayer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def forward_prop(self, prev_activations):
        if len(prev_activations[:, 0].shape) == 1:
            return prev_activations


class Conv2D:
    pass