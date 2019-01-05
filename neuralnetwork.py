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


class NeuralNetwork:
    class NNLayer:
        def __init__(self, prev_dims, num_neurons, g, g_backwards):
            self.numneurons = num_neurons
            self.W = np.random.randn(num_neurons, prev_dims) * np.sqrt(2 / prev_dims)
            self.b = np.random.randn(num_neurons, 1)
            self.g = g
            self.g_backwards = g_backwards
            # Z will be computed during forward prop
            self.prev_activations = None
            self.Z = None

        def forward_prop(self, prev_activations):
            Z = np.dot(self.W, prev_activations) + self.b
            self.prev_activations = prev_activations
            self.Z = Z
            return self.g(Z)


    def __init__(self, layer_dims):
        self.layers = []
        self.dims = layer_dims
        for i in range(1, len(layer_dims)):
            g = ActivationFunctions.relu
            g_backwards = ActivationFunctions.relu_backwards
            if i == len(layer_dims) - 1:
                g = ActivationFunctions.sigmoid
                g_backwards = ActivationFunctions.sigmoid_backwards
            self.layers.append(NeuralNetwork.NNLayer(layer_dims[i - 1], layer_dims[i],
                                                     g, g_backwards))

    def compute_forward_prop(self, X):
        self.m = X.shape[1]
        A_prev = X
        A = []
        for i in range(len(self.layers)):
            A = self.layers[i].forward_prop(A_prev)
            # print("\nLayer {} activations: \n{}".format(i, A))
            A_prev = A
        return A[0]

    def Loss(self, y, yhat):
        y = y.reshape(self.m)
        # print(y)
        # print(yhat)
        return -(y*np.log(yhat) + (1-y)*np.log(1-yhat))

    def dLdA(self, A, Y):
        return -Y / A + (1 - Y) / (1 - A)

    def J(self, Y, Yhat):
        losses = self.Loss(Y, Yhat)
        return np.sum(losses) / losses.shape[0]

    def _backprop(self, A, Y, alpha):
        dA = self.dLdA(A, Y)
        for i in range(len(self.layers) - 1, -1, -1):
            curr_layer = self.layers[i]
            dZ = dA * curr_layer.g_backwards(curr_layer.Z)
            dW = (1. / self.m) * np.dot(dZ, curr_layer.prev_activations.T)
            db = (1. / self.m) * np.sum(dZ, axis=1, keepdims = True)
            curr_layer.W -= alpha * dW
            curr_layer.b -= alpha * db
            dA = np.dot(curr_layer.W.T, dZ)

    def train(self, X, Y, alpha=0.01, num_iters=500, convergence_thresh=0.0):
        costs = []
        for i in range(num_iters):
            Yhat = self.compute_forward_prop(X)
            cost = self.J(Y, Yhat)
            if i % 100 == 0:
                print("Cost at iteration {}: {}".format(i, cost))
            costs.append(cost)

            if i > 1 and abs(costs[i-1] - costs[i]) < convergence_thresh:
                break

            self._backprop(Yhat, Y, alpha=alpha)
        return costs

    def predict(self, X):
        return (self.compute_forward_prop(X) > 0.5)

    def get_accuracy(self, X, Y):
        print("Expected:    ", Y)
        Yhat = self.predict(X)
        print("Predictions: ", Yhat)
        matches = (Yhat == Y.reshape(self.m))
        print(matches)
        return np.sum(matches) / len(matches)


if __name__ == "__main__":
    nx = 10
    m = 12
    layer_dims = [nx, 20, 30, 50, 70, 6, 1]
    nn = NeuralNetwork(layer_dims)
    X = np.array(np.random.randn(nx, 12))
    Y = np.array([[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]])
    costs = nn.train(X, Y, alpha=0.01)
    accuracy_on_train = nn.get_accuracy(X, Y)
    print(accuracy_on_train)

