import numpy as np


class Perceptron:
    def __init__(self, inputs, hidden, outputs):
        self.w1 = np.random.randn(inputs, hidden)
        self.w2 = np.random.randn(hidden, outputs)
        self.b1 = np.zeros((1, hidden))
        self.b2 = np.zeros((1, outputs))

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_deriv(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        res = self.activation(self.z2)
        return res

    def backward_prop(self, X, y, y_hat, LR):
        error = y - y_hat
        delta2 = error * self.activation_deriv(y_hat)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = np.dot(delta2, self.w2.T) * self.activation_deriv(self.a1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        self.w1 += LR * dW1
        self.b1 += LR * db1
        self.w2 += LR * dW2
        self.b2 += LR * db2

    def learn(self, X, y, LR=0.1, epoch=100):
        for i in range(epoch):
            y_hat = self.forward(X)
            self.backward_prop(X, y, y_hat, LR)
            if i % 100 == 0:
                loss = np.mean((y-y_hat) ** 2)
                print(f"Iteration {i}: loss = {loss}")

    def predict(self, X):
        y_hat = self.forward(X)
        return y_hat

