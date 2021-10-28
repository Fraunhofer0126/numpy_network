import numpy as np

def activation_func(type):
    def sigmoid(x, deriv):
        if deriv == True:
            return 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))
        else:
            return 1 / (1 + np.exp(-x))

    def relu(x, deriv):
        if deriv == True:
            return None
        else:
            return np.maximum(0, x)


    if type == 'sigmoid':
        return sigmoid
    elif type == 'relu':
        return relu

def cost_func(type):
    def mse(prediction, label, deriv):
        if deriv == False:
            return 1/2 * np.sum((prediction - label)**2)
        else:
            return prediction - label
    if type == 'mse':
        return mse


def accuracy(a, y):
    mini_batch = a.shape[1]
    idx_a = np.argmax(a, axis=0)
    idx_y = np.argmax(y, axis=0)
    acc = sum(idx_a == idx_y) / mini_batch
    return acc

