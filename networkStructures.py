# coding by kangan
# 11.7

import numpy as np
from math import sqrt, ceil
from abc import ABC, abstractmethod
from utils import col2im, im2col


class Layer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def bp(self, delta):
        pass

    @abstractmethod
    def update(self):
        pass

class Sigmoid(Layer):
    def __init__(self):
        self.input = None
        self.output = None
        self.name = 'sigmoid'

    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def bp(self, delta):
        return delta*self.output*(1-self.output)

    def update(self):
        pass

class ReLU(Layer):
    def __init__(self):
        self.input = None
        self.output = None
        self.name = 'relu'

    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output

    def bp(self, delta):
        delta[self.input < 0] = 0
        return delta

    def update(self):
        pass



class Dense(Layer):
    def __init__(self, outputSize, optim):
        self.input = None
        self.output = None
        self.inputSize = None
        self.outputSize = outputSize
        self.bias = np.zeros((outputSize, 1))
        self.optim = optim
        self.gradw = None
        self.gradbias = None
        self.optimizer = optim
        self.initw = False
        self.name = 'dense'

    def forward(self, x):
        # ka: You don't need to Enter the 'inputSize' in __init__() :)
        if self.initw == False:
            self.inputSize = x.shape[0]
            self.w = np.random.randn(self.outputSize, self.inputSize) * sqrt(6 / (self.inputSize + self.outputSize))
            self.initw = True

        self.input = x
        self.output = np.dot(self.w, x) + self.bias
        return self.output

    def bp(self, delta_next):
        batchSize = delta_next.shape[1]
        self.gradw = np.dot(delta_next, self.input.T) / batchSize
        self.gradbias = (np.sum(delta_next, axis=1) / batchSize).reshape(-1,1)

        delta = np.dot(self.w.T, delta_next)
        return delta

    def update(self):
        self.w += self.optimizer(self.gradw, weight=True)
        self.bias += self.optimizer(self.gradbias, weight=False)


class Conv(Layer):
    def __init__(self, ksize, pad, stride, channel, filters, optim):
        self.kernelSize = ksize
        self.padding = pad
        self.stride = stride
        self.Channel = channel
        # ka: filter shape:  (output_channel, input_channel, height, width) or (FN, C, FH, FW)
        self.w = np.random.randn(filters, channel, ksize, ksize) * sqrt(6 /(filters*channel* (ksize + ksize)))
        self.bias = np.zeros(filters)
        self.input = None
        self.output = None
        self.optimizer = optim


    def forward(self, x):
        FN, C, FH, FW = self.w.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.padding - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.padding - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.w.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.bias
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def bp(self, dout):
        FN, C, FH, FW = self.w.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)

        self.dW = np.dot(self.col.T, dout)
        self.dW = np.transpose(self.dW, (1,0))
        self.dW = self.dW.reshape((FN, C, FH, FW))

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.padding)

        return dx

    def update(self):
        self.w += self.optimizer(self.dW, weight=True)
        self.bias += self.optimizer(self.db, weight=False)

class Pooling(Layer):
    """
    MaxPooling
    1. No param need learning
    2. Channels won't change

    """
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.input = None
        self.output = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # 展开
        col= im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        # 最大值
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        # 转换
        out = np.reshape(out, (N, out_h, out_w, C))
        out = np.transpose(out, (0, 3, 1, 2))

        self.input = x
        self.output = out
        self.arg_max = arg_max

        return out

    def bp(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.input.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

    def update(self):
        pass

class Flatten(Layer):
    def __init__(self, batchSize):
        self.inputSize = None
        self.outputSize = batchSize
        self.input = None
        self.output = None

    def forward(self, x):
        self.inputSize = x.shape
        output = x.reshape((-1, self.outputSize))
        self.output = output
        return output

    def bp(self, delta_next):
        delta = delta_next.reshape(self.inputSize)
        return delta

    def update(self):
        pass


