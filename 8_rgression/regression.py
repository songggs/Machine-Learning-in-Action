# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:38:33 2017

@author: yang
"""

import numpy as np
import matplotlib.pyplot as plt


"""""""""

回归

"""""""""
'''
导入数据
'''
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


'''
普通最小二乘法求w
w = [(xTX)-1]XTy
'''
def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('这个矩阵是非奇异的，不能求逆')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws





"""""""""

局部加权线性回归

"""""""""
'''
给定测试点和数据集，返回测试点的结果
para:
    testPonit：测试点, 
    xArr：数据集, 
    yArr：标签, 
    k：衰减权重   
'''
def lwlr(testPonit, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPonit - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('这个矩阵是非奇异的，不能求逆')
        return
    ws = xTx.I * (xMat.T * weights *yMat)
    return testPonit * ws


'''
给定测试数据集，返回预测结果
'''
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


'''
误差
'''
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()





"""""""""

岭回归

"""""""""
'''
给定数据集、标签和lamda，得到回归系数ws
'''
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    demon = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(demon) == 0.0:
        print('奇异矩阵，不能求逆')
        return
    ws = demon.I * (xMat.T * yMat)
    return ws


'''
求一系列lamda对应的ws
'''
def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat-xMean)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat





"""""""""

前向逐步回归

"""""""""
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean)/xVar
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        #print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE<lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat
                
                
                
                
                
"""""""""

预测乐高玩具价格

"""""""""              
'''
购物信息的获取
从页面读取数据，生成retX和retY列表
Parameters:
    retX - 数据X
    retY - 数据Y
    inFile - HTML文件
    yr - 年份
    numPce - 乐高部件数目
    origPrc - 原价
'''
# -*-coding:utf-8 -*-
from bs4 import BeautifulSoup
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r = "%d" % i)
    while(len(currentRow) != 0):
        currentRow = soup.find_all('table', r = "%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r = "%d" % i)


'''
依次读取六种乐高套装的数据，并生成数据矩阵
'''
def setDataCollect(retX, retY):
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99) #2006年的乐高828
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99) #2002年的乐高
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99) #2007年的乐高
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99) #2007年的乐高
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99) #2008年的乐高
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)


'''
交叉验证测试岭回归
'''
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        np.random.shuffle(indexList)
        for j in range(m):
            if j<m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0)
            varTrain = np.var(matTrainX, 0)
            matTestX = (matTestX-meanTrain)/varTrain
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
    meanErrors = np.mean(errorMat,0) #计算每次交
    minMean = float(min(meanErrors)) #找到最小误
    bestWeights = wMat[np.nonzero(meanErrors == minMean)] #找到最佳回
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights / varX
    print(-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat))
    
    
    


if __name__ == '__main__':
    
    print('------最小二乘法求ws------')
    xArr, yArr = loadDataSet(r'F:\Python\rgression\ex0.txt')
    ws = standRegres(xArr, yArr)
    print(ws)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    
    
    print('-------画出点图-------')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    plt.show()
    
    print('------画出拟合线-------')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat, c='g')
    plt.show()

    
    print('-------局部加权线性回归-------')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    yHat = lwlrTest(xCopy, xArr, yArr, k=0.01)    
    ax.plot(xCopy[:, 1], yHat, c='r')
    plt.show()
    
    print('-------预测鲍鱼年龄-------')
    abX, abY = loadDataSet(r'F:\Python\rgression\abalone.txt')
    yHat01 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 0.1)
    yHat011 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 0.1)
    error = rssError(abY[0:99], yHat01)
    error1 = rssError(abY[100: 199], yHat011)
    print('k=0.1,训练集误差：', error)
    print('k=0.1,测试集误差：', error1)
    
    yHat1 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 1.0)
    error = rssError(abY[0:99], yHat1)
    yHat11 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 1.0)
    error1 = rssError(abY[100:199], yHat1)
    print('k=1,训练集误差：', error)
    print('k=1,测试集误差：', error1)
    
    yHat10 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 10)
    error = rssError(abY[0:99], yHat10)
    yHat101 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 10)
    error1 = rssError(abY[100:199], yHat10)
    print('k=10,训练集误差：', error)
    print('k=10,测试集误差：', error1)

    print('-------岭回归-------')
    abX, abY = loadDataSet(r'F:\Python\rgression\abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

    print('-------前向逐步回归-------')
    returnMat = stageWise(abX, abY, 0.005, 5000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    plt.show()
   
    print('-------乐高积木-------')
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    m, n = np.shape(lgX)
    lgX1 = np.mat(np.ones((63,5)))
    lgX1[:, 1:5] = np.mat(lgX)
    ws = standRegres(lgX1, lgY)
    print('%f + %f * Year + %f * Numpieces + %f * NewOrUsed + %f * original price' % (ws[0], ws[1], ws[2], ws[3], ws[4]))
    
    crossValidation(lgX, lgY)
    
    
    
    
    
    
    
    
