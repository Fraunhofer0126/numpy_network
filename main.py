# coding by kangan
# 10.28

import numpy as np
from math import sqrt, ceil
from utils import activation_func, accuracy, cost_func
from abc import ABC, abstractmethod
from scipy.io import loadmat
from network import Network
from networkStructures import Dense

def loadSVHN():
    m_train = loadmat('./train_32x32.mat')
    x_train_raw = m_train['X']
    x_train = x_train_raw.reshape((32 * 32 * 3, -1)) / 255
    label_train_raw = m_train['y']

    label_train = np.zeros((10, label_train_raw.shape[0]))

    for index in range(label_train_raw.shape[0]):  # 73257
        if label_train_raw[index][0] == 10:
            label_train[0][index] = 1
        else:
            label_train[label_train_raw[index][0]][index] = 1

    m_test = loadmat('./test_32x32.mat')
    x_test_raw = m_test['X']
    x_test = x_test_raw.reshape((32 * 32 * 3, -1)) / 255
    label_test_raw = m_test['y']

    label_test = np.zeros((10, label_test_raw.shape[0]))

    for index in range(label_test_raw.shape[0]):  # 73257
        if label_test_raw[index][0] == 10:
            label_test[0][index] = 1
        else:
            label_test[label_test_raw[index][0]][index] = 1

    return x_train, label_train, x_test, label_test

def loadMnist():
    m = loadmat("./mnist_small_matlab.mat")

    trainData, label_train = m['trainData'], m['trainLabels']
    testData, label_test = m['testData'], m['testLabels']

    train_size = 10000
    x_train = trainData.reshape(-1, train_size)
    test_size = 2000
    x_test = testData.reshape(-1, test_size)
    return x_train, label_train, x_test, label_test

if __name__ == '__main__':


    x_train, label_train, x_test, label_test = loadMnist()
    model = Network('mse')
    model.Sequential([
        Dense(inputSize=28*28, outputSize=256, act='sigmoid'),
        Dense(inputSize=256, outputSize=128, act='sigmoid'),
        Dense(inputSize=128, outputSize=64, act='sigmoid'),
        Dense(inputSize=64, outputSize=10, act='sigmoid'),
    ])

    model.train(lr = 0.005, epochs=200, batchSize=100, trainX=x_train, trainY=label_train, testX=x_test, testY = label_test)