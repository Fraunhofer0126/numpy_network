# coding by kangan
# 10.28
# to do
# 1. 动量/自动 梯度
# 2. Conv 正反向传播

# 3. Average Pooling 正反向
# 4. 激活函数层
# 5.


import numpy as np
from math import sqrt, ceil
from utils import activation_func, accuracy, cost_func
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

class Dense(Layer):

    #        |-------------Layer------------|                 |-----------layer_next---------...
    # input->|--->w* ---> z ----> act --->a-|---input_next--->|--...
    def __init__(self, inputSize, outputSize, act):
        self._inputSize = inputSize
        self._outputSize = outputSize
        self.input = None
        self.a = None
        self.z = None
        self.w = np.random.randn(outputSize, inputSize)*sqrt(6/(inputSize+outputSize))
        self.delta = None
        self.activation = activation_func(act)

    def __call__(self, **kwargs):
        if kwargs['forward']:
            self.input = kwargs['forward_input']
            self.z = np.dot(self.w, self.input)
            self.a = self.activation(self.z, deriv=False)
            return self.a
        else:
            _isFinalLayer = kwargs['isFinalLayer']
            if _isFinalLayer:
                self.delta = kwargs['cost_func'](kwargs['prediction'], kwargs['label'], deriv = True) * self.activation(self.z, deriv = True)
                _isFinalLayer = False
            else:
                self.delta = np.dot(kwargs['w_next'].T, kwargs['bp_delta']) * self.activation(self.z, deriv=True)
            grad_w = np.dot(self.delta, self.input.T)
            self.w -= kwargs['lr'] * grad_w
            return self.delta, _isFinalLayer, self.w

class Conv(Layer):

    # @param: padding_type 两种类型 'same', 'valid'

    def __init__(self, kernelSize, inChannel, outChannel, padding_type, stride, act):
        self._kernelSize = kernelSize
        self._inputSize = inChannel
        self._outputSize = outChannel
        self._padding = padding_type
        self._stride = stride
        self.activation = activation_func(act)
        weights_scale = sqrt(kernelSize * kernelSize * inChannel / 2)
        self.w = np.random.standard_normal((kernelSize, kernelSize, inChannel, outChannel)) / weights_scale
        self.input = None
    # to do
    # np.pad()
    def __call__(self, **kwargs):
        pass







