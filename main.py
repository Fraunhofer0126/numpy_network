# coding by kangan
# 10.29

import numpy as np
from math import sqrt, ceil
from utils import accuracy, cost_func, optim_momentum
from scipy.io import loadmat
from network import Network
from networkStructures import Dense, Sigmoid, Conv, Pooling, Flatten, ReLU

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


def loadSVHN_conv():
    m_train = loadmat('./train_32x32.mat')
    x_train_raw = m_train['X']
    x_train = np.transpose(x_train_raw, (3, 2, 0, 1))
    label_train_raw = m_train['y']
    print(label_train_raw.shape)
    label_train = np.zeros((10, label_train_raw.shape[0]))

    for index in range(label_train_raw.shape[0]):  # 73257
        if label_train_raw[index][0] == 10:
            label_train[0][index] = 1
        else:
            label_train[label_train_raw[index][0]][index] = 1

    m_test = loadmat('./test_32x32.mat')
    x_test_raw = m_test['X']
    x_test = np.transpose(x_test_raw, (3, 2, 0, 1))
    label_test_raw = m_test['y']

    label_test = np.zeros((10, label_test_raw.shape[0]))

    for index in range(label_test_raw.shape[0]):  # 73257
        if label_test_raw[index][0] == 10:
            label_test[0][index] = 1
        else:
            label_test[label_test_raw[index][0]][index] = 1

    return x_train, label_train, x_test, label_test

def loadMnist_conv():
    m = loadmat("./mnist_small_matlab.mat")

    trainData, label_train = m['trainData'], m['trainLabels']
    testData, label_test = m['testData'], m['testLabels']

    print(trainData.shape)
    x_train = []
    x_test = []
    for k in range(trainData.shape[2]):
        img = trainData[:,:,k].reshape((1, 28, 28))
        x_train.append(img)
    for k in range(testData.shape[2]):
        img = testData[:,:,k].reshape((1, 28, 28))
        x_test.append(img)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    return x_train, label_train, x_test, label_test

if __name__ == '__main__':

    model = Network('mse')
    batchSize = 100
    #
    model.Sequential([
        Dense(outputSize=512, optim=optim_momentum()),
        Sigmoid(),
        Dense(outputSize=256, optim = optim_momentum()),
        Sigmoid(),
        Dense(outputSize=128, optim = optim_momentum()),
        Sigmoid(),
        Dense(outputSize=64, optim = optim_momentum()),
        Sigmoid(),
        Dense(outputSize=10, optim = optim_momentum()),
        Sigmoid(),
    ])


    # Conv: (ksize, pad, stride, channel, filters, optim)
    # Pooling: (pool_h, pool_w, stride=1, pad=0)
    # model.Sequential([
    #     Conv(5, 0, 1, 1, 2, optim = optim_momentum()),
    #     Sigmoid(),
    #     Pooling(12, 12, 1, 0),
    #     Conv(5, 0, 1, 2, 4, optim = optim_momentum()),
    #     Sigmoid(),
    #     Flatten(batchSize),
    #     Dense(outputSize=64, optim = optim_momentum()),
    #     Sigmoid(),
    #     Dense(outputSize=10, optim=optim_momentum()),
    #     Sigmoid()
    # ])


    x_train, label_train, x_test, label_test = loadSVHN()
    print(x_train.shape, label_train.shape, x_test.shape, label_test.shape)

    # You can set do_you_want_to_train = False to skip Train and test directly!!

    do_you_want_to_train = False

    if do_you_want_to_train == True:
        model.train(epochs=200,batchSize=batchSize,trainX=x_train,trainY=label_train,testX=x_test,testY=label_test,modelname = "model.pkl")
        model.saveWeights("model_final.pkl")

    # test!
    model.test("model_final.pkl", x_test, label_test)