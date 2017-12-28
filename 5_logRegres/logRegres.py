# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:13:59 2017

@author: yang
"""
import numpy as np
import matplotlib.pyplot as plt

"""""""""

普通逻辑回归算法

"""""""""
'''
加载数据集，返回数据集和标签
其中数据集shape为m*n，标签的shape为1*m
'''
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open(r'F:\Python\logRegres\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


'''
定义S函数
'''
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


'''
梯度上升算法，给定数据集和标签，返回权重
'''
def gradAscent(dataMatIn, classLabels):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weight = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMat*weight)
        error = (labelMat-h)
        weight = weight + alpha * dataMat.transpose() * error
    return weight


'''
画出决策边界
'''
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2,ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.xlabel('X2')
    plt.show()





"""""""""

随机梯度上升法

"""""""""
def stocGradAscent(dataMat, classMat, numIter=150):
    dataMat = np.array(dataMat)
    m,n = np.shape(dataMat)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex]*weights))
            error = classMat[randIndex] - h
            weights = weights + alpha*error*dataMat[randIndex]
            del(dataIndex[randIndex])
    return weights
            

"""""""""

预测病马死亡率

"""""""""
'''
给定测试数据和权重，返回分类
'''
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else: 
        return 0.0
    
    
'''
测试
'''
def colicTest():
    frTrain = open(r'F:\Python\logRegres\horseColicTraining.txt')
    frTest = open(r'F:\Python\logRegres\horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent(trainingSet, trainingLabels, 500)
    
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(lineArr,trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('错误率为：%f' % errorRate)    
    return errorRate


'''
训练10次，求平均误差
'''
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is:%f' % (numTests,(errorSum/float(numTests))))





if __name__ == '__main__':
    
    print('-------梯度上升法-------')
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    print(weights)
    
    plotBestFit(weights.getA())
    
    print('-------随机梯度上升法-------')
    weights = stocGradAscent(dataMat, labelMat)
    print(weights)
    plotBestFit(weights)
    
    print('-------病马死亡率预测-------')
    multiTest()
    
    
    
    