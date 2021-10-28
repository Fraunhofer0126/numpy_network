# coding by kangan
# 10.28

import numpy as np
from math import sqrt, ceil
from utils import activation_func, accuracy, cost_func


class Network:
    def __init__(self, cost_func_name):
        self.layerList = []
        self.cost_func = cost_func(cost_func_name)

    def Sequential(self, layerList):
        self.layerList = layerList

    def predict(self, input):
        output = None
        for layer in self.layerList:
            output = layer(forward = True, forward_input = input)
            input = output
        return output

    def bp(self, prediction, label, lr):
        delta = None
        isFinalLayer = True
        w_next = None
        for index in range(len(self.layerList)-1, -1, -1):
            delta, isFinalLayer, w_next = self.layerList[index](forward = False,isFinalLayer = isFinalLayer,prediction = prediction,label = label,bp_delta = delta,cost_func = self.cost_func,lr = lr, w_next = w_next)

    def train(self, lr, epochs, batchSize, trainX, trainY, testX, testY):
        trainSize = trainX.shape[1]
        testSize = testX.shape[1]
        for epoch in range(epochs):
            train_loss = 0
            train_acc = 0
            test_loss = 0
            test_loss = 0
            idxs = np.random.permutation(trainSize)
            for k in range(ceil(trainSize / batchSize)):
                start_idx = k * batchSize
                end_idx = min((k + 1) * batchSize, trainSize)

                batch_indices = idxs[start_idx:end_idx]
                input = trainX[:, batch_indices]
                y = trainY[:, batch_indices]

                result = self.predict(input = input)
                self.bp(prediction = result, label = y, lr = lr)

                train_loss += self.cost_func(result, y, deriv = False) / batchSize
                train_acc += accuracy(result, y)

            train_loss /= ceil(trainSize / batchSize)
            train_acc /= ceil(trainSize / batchSize)
            test_result = self.predict(input = testX)
            test_loss = self.cost_func(test_result, testY, deriv = False) / testSize
            test_acc = accuracy(test_result, testY)
            print(">>epoch {}: train_loss = {:.4f}, train_acc = {:.4f}, test_loss = {:.4f}, test_acc = {:.4f}".format(
                epoch, train_loss, train_acc, test_loss, test_acc))