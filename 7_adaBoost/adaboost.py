# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:13:32 2017

@author: yang
"""

import numpy as np
import matplotlib.pyplot as plt

"""""""""

单层决策树弱分类器的adaBoost训练过程

"""""""""

'''
构建数据
'''
def loadSimpData():
    datMat = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.],[2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


'''
根据特征、阈值、大小于号判断类别
para:
    dataMat:输入数据, 
    dimen：特征所在列，即对每一个特征, 
    threshVal：阈值, 
    threshIneq：切换阈值方向，即大于号或小于号
return：
    分类结果向量
'''
def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMat[:, dimen] <= threshVal] = -1.0
    else:        
        retArray[dataMat[:, dimen] > threshVal] = -1.0
    return retArray
    

'''
单层决策树生成
para:
    dataArr:数据集, 
    classLabels：标签, 
    D：每个样本的权重向量
return：
    bestStump:包含特征索引，阈值，大小号的最佳单层决策树，是一个弱分类器, 
    minError：最佳分类器的最小误差, 
    bestStump：最佳分类器的预测结果标签
'''    
def buildStump(dataArr, classLabels, D):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n =np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMat[:,i].min()
        rangeMax = dataMat[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictVals = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictVals == labelMat] = 0
                weightError = D.T * errArr
                print('dim: %d, thresh: %.2f, thresh inequal: %s, error: %.3f' % (i, threshVal, inequal, weightError))
                
                if weightError < minError:
                    minError = weightError
                    bestClasEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst
    
    
'''
单层决策树的adaBoost
'''    
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = list(np.shape(dataArr))[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, minError, bestClasEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5*np.log((1.0-minError)/max(minError, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:", bestClasEst.T)
        
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, bestClasEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * bestClasEst
        print("aggClassEst:", aggClassEst)
        aggError = np.multiply(np.sign(aggClassEst) !=np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggError.sum()/m
        print("errorRate", errorRate)
        if errorRate==0.0:
            break
    return weakClassArr,aggClassEst
    
    
'''
用adaBoost分类，即测试
para:
    datToClass:数据, 
    classifierArr:弱分类器的adaBoost
'''   
def adaClassify(datToClass, classifierArr):
    dataMat = np.mat(datToClass)
    m = list(np.shape(dataMat))[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['threshVal'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)
    

'''
在另一个数据集上训练并测试
自适应数据加载函数
'''
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


    


"""""""""

绘制ROC曲线，计算AUC函数

"""""""""
def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = np.sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1], [0,1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title('ROC')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('AUC:', ySum*xStep)








if __name__ == '__main__':
    datMat, classLabels = loadSimpData()
    print(datMat, classLabels)
    D = np.mat(np.ones((5, 1))/5)
    print('-------弱分类器-------')
    buildStump(datMat, classLabels, D)
    print('-------adaBoost-------')
    classifierArr, aggClassEst = adaBoostTrainDS(datMat, classLabels, numIt = 40)
    print('-------测试-------')
    ans = adaClassify([0,0], classifierArr)
    print(ans)
    
    print('=====================')
    print('-------新数据集上训练、测试-------')
    dataMat0, labelMat0 = loadDataSet(r'F:\Python\adaBoost\horseColicTraining2.txt')
    classify, aggClassEst = adaBoostTrainDS(dataMat0, labelMat0, numIt = 10)
    
    dataMat1, labelMat1 = loadDataSet(r'F:\Python\adaBoost\horseColicTest2.txt')
    predict  = adaClassify(dataMat1, classify)
    errArr = np.mat(np.ones((67, 1)))
    error = errArr[predict != np.mat(labelMat1).T].sum()
    print(error)
    
    print('-------ROC曲线-------')
    plotROC(aggClassEst.T, labelMat0)
    
    