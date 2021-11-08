# coding by kangan
# 11.7
import pickle
import numpy as np
from math import sqrt, ceil
from utils import accuracy, cost_func
import matplotlib.pyplot as plt

class Network:
    def __init__(self, cost_func_name):
        self.layerList = []
        self.cost_func = cost_func(cost_func_name)
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

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

    def show(self, epochs):
        x =range(0, epochs, 1)
        plt.title('Loss')
        plt.plot(x, self.train_loss, color='red', marker='o', label='train loss')
        plt.plot(x, self.test_loss, color='blue', marker='v', label='test loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

        plt.title('Accuracy')
        plt.plot(x, self.train_acc, color='red', marker='o', label='train acc')
        plt.plot(x, self.test_acc, color='blue', marker='v', label='test acc')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.show()

    def saveWeights(self, modelname):
        wList = []
        for layer in self.layerList:
            if hasattr(layer, 'w'):
                wList.append(layer.w)
            else:
                wList.append(None)
        with open(modelname, 'wb') as f:  # 将数据写入pkl文件
            pickle.dump(wList, f)

    def train(self, epochs, batchSize, trainX, trainY, testX, testY):
        print("Start Training!")

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

            testX_tmp = testX
            testY_tmp = testY

            test_result = self.forward(testX_tmp)
            test_loss = self.cost_func(test_result, testY_tmp, deriv = False) / testSize
            test_acc = accuracy(test_result, testY_tmp)

            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)
            self.test_loss.append(test_loss)
            self.test_acc.append(test_acc)

            print(">>epoch {}: train_loss = {:.4f}, train_acc = {:.4f}, test_loss = {:.4f}, test_acc = {:.4f}".format(
                epoch, train_loss, train_acc, test_loss, test_acc))

        self.show(epochs)

    def test(self, weightName, testX, testY):
        with open(weightName, 'rb') as f:  # 读取pkl文件数据
            wList = pickle.load(f)
        for index in range(len(self.layerList)):
            self.layerList[index].w = wList[index]

        testSize = testX.shape[0]

        test_result = self.forward(testX)
        test_loss = self.cost_func(test_result, testY, deriv=False) / testSize
        test_acc = accuracy(test_result, testY)
        print("Final: test_loss = {:.4f}, test_acc = {:.4f}".format(test_loss, test_acc))

