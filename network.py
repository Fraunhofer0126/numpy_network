# coding by kangan
# 11.7
import pickle
import numpy as np
from math import sqrt, ceil
from utils import accuracy, cost_func
import matplotlib.pyplot as plt
from networkStructures import *

class Network:
    def __init__(self, cost_func_name):
        self.layerList = []
        self.cost_func_name = cost_func_name
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

    def saveModels(self, modelname):
        model_info_dict = {}
        wList = []
        layerList = []
        optimList = []
        outputSizeList = []
        for layer in self.layerList:
            if hasattr(layer, 'w'):
                wList.append(layer.w)
            else:
                wList.append(None)

            layerList.append(layer.name)

            if hasattr(layer, 'outputSize'):
                outputSizeList.append(layer.outputSize)
            else:
                outputSizeList.append(None)

            if hasattr(layer, 'optimizer'):
                optimList.append(layer.optimizer.name)
            else:
                optimList.append(None)



        model_info_dict['wList'] = wList
        model_info_dict['layerList'] = layerList
        model_info_dict['optimList'] = optimList
        model_info_dict['outputSizeList'] = outputSizeList
        model_info_dict['cost_func_name'] = self.cost_func_name

        with open(modelname, 'wb') as f:  # 将数据写入pkl文件
            pickle.dump(model_info_dict, f)
            print("Model has been saved to: ", modelname)

    def loadModels(self, modelname):
        with open(modelname, 'rb') as f:  # 读取pkl文件数据
            model_info_dict = pickle.load(f)
            print("Loaded the model: ", modelname)

        wList = model_info_dict['wList']
        layerList = model_info_dict['layerList']
        optimList = model_info_dict['optimList']
        outputSizeList = model_info_dict['outputSizeList']
        cost_func_name = model_info_dict['cost_func_name']

        self.layerList = []
        for index in range(len(layerList)):
            if layerList[index] == 'relu':
                self.layerList.append(ReLU())
            elif layerList[index] == 'sigmoid':
                self.layerList.append(Sigmoid())
            elif layerList[index] == 'dense':
                self.layerList.append(Dense(outputSize=outputSizeList[index], optim=optimList[index]))

        for index in range(len(self.layerList)):
            if hasattr(self.layerList[index], 'initw'):
                self.layerList[index].initw = True
            self.layerList[index].w = wList[index]
            self.layerList[index].optimizer = optimList[index]
        self.cost_func_name = cost_func_name


    def train(self, epochs, batchSize, trainX, trainY, testX, testY, modelname):
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

            if epoch >= 10 and epoch % 10 == 0:
                self.saveModels(modelname = modelname)

        self.show(epochs)

    def test(self, modelname, testX, testY):
        self.loadModels(modelname)

        testSize = testX.shape[0]

        test_result = self.forward(testX)
        test_loss = self.cost_func(test_result, testY, deriv=False) / testSize
        test_acc = accuracy(test_result, testY)
        print("Final: test_loss = {:.4f}, test_acc = {:.4f}".format(test_loss, test_acc))

