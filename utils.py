# coding by kangan
# 11.7

import numpy as np

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

class optim_momentum:
    def __init__(self, learning_rate=0.5, eps=0.8):
        self.eps = eps
        self.v_weight = np.zeros(1)
        self.v_bias = np.zeros(1)
        self.learning_rate = learning_rate
    def __call__(self, grad, weight=True):
        if weight:
            self.v_weight = self.eps * self.v_weight - self.learning_rate * grad
            return self.v_weight
        else:
            self.v_bias = self.eps * self.v_bias - self.learning_rate * grad
            return self.v_bias

def im2col(input_data, filter_h, filter_w, stride, pad):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = np.transpose(col, (0, 4, 5, 1, 2, 3))
    col = col.reshape((N*out_h*out_w, -1))
    return col

def col2im(col, input_shape, filter_h, filter_w, stride, pad):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
