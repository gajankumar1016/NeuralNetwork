import numpy as np
import math
from utils import ActivationFunctions
import layers

class NeuralNetwork:
    def __init__(self):
        self.regularization = None
        self.layers = []

    def add_layer(self, layer):
        if type(layer) == layers.Flatten:
            self.layers.append(layer)
            return

        if len(self.layers) >= 1:
            prev_dims = self.layers[-1].num_neurons
            layer._initialize(prev_dims)
            self.layers.append(layer)


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
        return -(y*np.log(yhat) + (1-y)*np.log(1-yhat))

    def dLdA(self, A, Y):
        return -Y / A + (1 - Y) / (1 - A)

    def J(self, Y, Yhat):
        losses = self.Loss(Y, Yhat)
        cross_entropy_cost = np.sum(losses) / losses.shape[0]
        if self.regularization == "L2":
            L2_regularization_cost = 0
            for i in range(len(self.layers)):
                L2_regularization_cost += np.sum(self.layers[i].W**2)

            L2_regularization_cost *= (self.lambd/(2 * self.m))
            return cross_entropy_cost + L2_regularization_cost
        return cross_entropy_cost

    def _backprop(self, A, Y, alpha):
        dA = self.dLdA(A, Y)
        for i in range(len(self.layers) - 1, -1, -1):
            curr_layer = self.layers[i]
            # TODO: handle case when layers can come before flatten
            if type(curr_layer) == layers.Flatten:
                continue
            dZ = dA * curr_layer.g_backwards(curr_layer.Z)
            dW = (1. / self.m) * np.dot(dZ, curr_layer.prev_activations.T)
            if self.regularization == "L2":
                dW += (self.lambd/self.m) * curr_layer.W
            db = (1. / self.m) * np.sum(dZ, axis=1, keepdims = True)
            curr_layer.W -= alpha * dW
            curr_layer.b -= alpha * db
            dA = np.dot(curr_layer.W.T, dZ)

    def _get_minibatches(self, X, Y, minibatch_size):
        m = X.shape[1]
        num_complete_minibatches = math.floor(m / minibatch_size)
        minibatches = []
        for i in range(num_complete_minibatches):
            minibatch_X = X[:, i*minibatch_size:(i+1)*minibatch_size]
            minibatch_Y = Y[:, i*minibatch_size:(i+1)*minibatch_size]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch)

        if m % minibatch_size != 0:
            minibatch_X = X[:, num_complete_minibatches * minibatch_size:]
            minibatch_Y = Y[:, num_complete_minibatches * minibatch_size:]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch)

        return minibatches


    def train(self, X, Y, alpha=0.01, num_epochs=1000, minibatch_size=32, convergence_thresh=0.0, cost_thresh=None,
              regularization=None, lambd=None, print_cost=False, print_interval=100, seed=None):
        if regularization:
            if regularization.upper() == "L2" and lambd:
                self.regularization = regularization
                self.lambd = lambd
            else:
                raise ValueError("Invalid regularization params")

        costs = []
        minibatches = self._get_minibatches(X, Y, minibatch_size)
        for i in range(num_epochs):
            cost = -1
            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                Yhat = self.compute_forward_prop(minibatch_X)
                cost = self.J(minibatch_Y, Yhat)
                self._backprop(Yhat, minibatch_Y, alpha=alpha)

            if cost_thresh and cost < cost_thresh:
                break

            if print_cost and  i % print_interval == 0:
                print("Cost after epoch {}: {}".format(i, cost))
                costs.append(cost)

                if len(costs) >= 2 and (costs[i//print_interval-1] - costs[i//print_interval]) < 0:
                    print("Cost went up!")
                    print("Costs: ", costs)
                    break
        return costs

    def predict(self, X):
        return (self.compute_forward_prop(X) > 0.5)

    def get_accuracy(self, X, Y):
        # print("Expected:    ", Y)
        Yhat = self.predict(X)
        # print("Predictions: ", Yhat)
        matches = (Yhat == Y.reshape(self.m))
        # print(matches)
        return np.sum(matches) / len(matches)


if __name__ == "__main__":
    nx = 10
    m = 12
    X = np.array(np.random.randn(nx, 12))
    Y = np.array([[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]])
    nn = NeuralNetwork()
    nn.add_layer(layers.Flatten(nx))
    nn.add_layer(layers.Dense(20, activation=ActivationFunctions.relu, initialization="He"))
    nn.add_layer(layers.Dense(30, activation=ActivationFunctions.relu, initialization="He"))
    nn.add_layer(layers.Dense(1, activation=ActivationFunctions.sigmoid, initialization="He"))

    costs = nn.train(X, Y, alpha=0.01, num_epochs=1500, print_cost=True)

    accuracy_on_train = nn.get_accuracy(X, Y)
    print(accuracy_on_train)

