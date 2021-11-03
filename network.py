# coding by kangan
# 10.29

import numpy as np
from math import sqrt, ceil
from utils import accuracy, cost_func


class Network:
    def __init__(self, cost_func_name):
        self.layerList = []
        self.cost_func = cost_func(cost_func_name)

    def Sequential(self, layerList):
        self.layerList = layerList

    def forward(self, x):
        for layer in self.layerList:
            x = layer.forward(x)
        return x

    def bp(self, delta):
        for layer in self.layerList[::-1]:
            delta = layer.bp(delta)

    def update(self):
        for layer in self.layerList:
            layer.update()

    def train(self, epochs, batchSize, trainX, trainY, testX, testY):
        print("Start Training!")
        # 需要改
        if trainX.ndim == 4:
            trainSize = trainX.shape[0]
            testSize = testX.shape[0]
        else:
            trainSize = trainX.shape[1]
            testSize = testX.shape[1]
        for epoch in range(epochs):
            train_loss = 0
            train_acc = 0
            test_loss = 0
            test_acc = 0
            idxs = np.random.permutation(trainSize)
            for k in range(ceil(trainSize / batchSize)):
                start_idx = k * batchSize
                end_idx = min((k + 1) * batchSize, trainSize)

                batch_indices = idxs[start_idx:end_idx]
                if trainX.ndim == 4:
                    input = trainX[batch_indices, :, :, :]
                else:
                    input = trainX[:, batch_indices]
                #input = trainX[:, batch_indices]
                y = trainY[:, batch_indices]

                result = self.forward(input)
                delta_final = self.cost_func(result, y, deriv=True)
                self.bp(delta_final)
                self.update()

                train_loss += self.cost_func(result, y, deriv = False) / batchSize
                train_acc += accuracy(result, y)


            train_loss /= ceil(trainSize / batchSize)
            train_acc /= ceil(trainSize / batchSize)


            #print("test!")
            # 随机切一个batchSize的test数据，这样不用使用全部数据集合
            idxs = np.random.permutation(testSize)
            if trainX.ndim == 4:
                testX_tmp = testX[idxs[:batchSize], :, :, :]
            else:
                testX_tmp = testX[:, idxs[:batchSize]]

            testY_tmp = testY[:, idxs[:batchSize]]

            test_result = self.forward(testX_tmp)
            test_loss = self.cost_func(test_result, testY_tmp, deriv = False) / testSize
            test_acc = accuracy(test_result, testY_tmp)
            print(">>epoch {}: train_loss = {:.4f}, train_acc = {:.4f}, test_loss = {:.4f}, test_acc = {:.4f}".format(
                epoch, train_loss, train_acc, test_loss, test_acc))