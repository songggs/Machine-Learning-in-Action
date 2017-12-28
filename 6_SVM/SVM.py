# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:00:21 2017

@author: Administrator
"""
import numpy as np
from os import listdir

"""""""""

简化版SMO算法，处理小规模数据

"""""""""

'''
加载文件，返回数据集和标签
'''
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
    
    
'''
随机选择alpha，返回下标
para:
    i：第一个alpha的下标
    m：数据总数
return：
    随机选择的alpha的下标
'''
def selectJrand(i, m):
    j = i
    while(j==i):
        j = int(np.random.uniform(0, m))
    return j
    
    
'''
调整alpha的取值范围
给定alpha的上边界和下边界，返回最后结果
'''
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

 
'''
实现简化版SMO算法
para：
    dataMatIng：数据集
    classLabels：标签类别
    C：常数，松弛变量
    toler：容错率
    maxIte：最大迭代次数
return：
    b和w
'''    
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m,n = np.shape(dataMat)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T*(dataMat*dataMat[i,:].T)) + b
            Ei = fXi - float(classLabels[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T*(dataMat*dataMat[j,:].T)) + b
                Ej = fXj - float(classLabels[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L==H:
                    print("L==H")
                    continue
                eta = 2.0*dataMat[i,:]*dataMat[j,:].T - dataMat[i,:]*dataMat[i,:].T - dataMat[j,:]*dataMat[j,:].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if  (np.abs(alphas[j]-alphaJold) < 0.0001):
                    print("J 改变过少")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i]-alphaIold) * dataMat[i,:]*dataMat[i,:].T - labelMat[j] * (alphas[j]-alphaJold) * dataMat[i,:]*dataMat[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i]-alphaIold) * dataMat[i,:]*dataMat[j,:].T - labelMat[j] * (alphas[j]-alphaJold) * dataMat[j,:]*dataMat[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
        if (alphaPairsChanged == 0):
            iter += 1
        else: 
            iter = 0
        print("迭代次数：%d" % iter)
    return b, alphas
                
                
                
            
            
"""""""""

完整版Platt SMO实现

"""""""""    
        
'''
构建数据结构，保存自定义值
'''        
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelsMat = classLabels
        self.C = C
        self.tol = toler
        self.m = list(np.shape(dataMatIn))[0]
        self.alpha = np.mat(np.zeros((self.m , 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m ,2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        
        
'''
给定索引值k，返回k的误差
'''
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alpha,oS.labelsMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelsMat[k])
    return Ek
    
    
'''
选择第二个alpha，以保证每次优化中，步长最大
返回alpha的下标索引和误差
'''    
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k==i:continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE 
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


'''
计算误差，存入缓存
'''
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


'''
完整版Platt SMO算法中的优化历程
给定第一个alpha，得到最优的第二个alpha的索引
'''
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelsMat[i]*Ei < -oS.tol) and (oS.alpha[i] < oS.C) or (oS.labelsMat[i]*Ei > oS.tol) and (oS.alpha[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alpha[i].copy()
        alphaJold = oS.alpha[j].copy()
        if(oS.labelsMat[i] != oS.labelsMat[j]):
            L = max(0, oS.alpha[j] - oS.alpha[i])
            H = min(oS.C, oS.C + oS.alpha[j] - oS.alpha[i])
        else:
            L = max(0, oS.alpha[j] + oS.alpha[i] - oS.C)
            H = min(oS.C, oS.alpha[j] + oS.alpha[i])        
        if L==H:
            print("L==H")
            return 0
        eta = 2.0*oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alpha[j] -= oS.labelsMat[j]*(Ei-Ej)/eta
        oS.alpha[j] = clipAlpha(oS.alpha[j], H, L)
        if (np.abs(oS.alpha[j]-alphaJold) < 0.0001):
            print("J 改变过少")
            return 0
        oS.alpha[i] += oS.labelsMat[j]*oS.labelsMat[i]*(alphaJold - oS.alpha[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelsMat[i] * (oS.alpha[i]-alphaIold) * oS.K[i,i] - oS.labelsMat[j] * (oS.alpha[j]-alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelsMat[i] * (oS.alpha[i]-alphaIold) * oS.K[i,j] - oS.labelsMat[j] * (oS.alpha[j]-alphaJold) * oS.K[j,j]
        if (0 < oS.alpha[i]) and (oS.C > oS.alpha[i]):
            oS.b = b1
        elif (0 < oS.alpha[j]) and (oS.C > oS.alpha[j]):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0
    
    
'''
完整版SMO外循环代码
'''
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alpha.A > 0) * (oS.alpha.A < oS.C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)   
    return oS.b, oS.alpha


'''
计算w
'''    
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)   
    labelMat = mat(classLabels).transpose()
    m, n = np.shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], X[i,:].T)
    return w





"""""""""

核函数

"""""""""

'''
定义核函数
para：
    X:数据集
    A：某一行数据
    kTup:核函数信息
return：
    核函数
'''
def kernelTrans(X, A, kTup):
    m, n =np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if(kTup[0] =='lin'):
        K = X * A.T
    elif(kTup[0] == 'rbf'):
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem:That Kernel is not recognized')
    return K


'''
测试
'''
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet(r'F:\Python\SVM\testSetRBF.txt')
    b, alphas = smoP(dataArr,labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs = datMat[svInd]
    labelSv = labelMat[svInd]
    print('there are %d Support Vectors' % np.shape(sVs)[0])
    m, n =np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSv, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('\n the training error rate is： %f' % ((float(errorCount)/m))*100)
    
    dataArr,labelArr = loadDataSet(r'F:\Python\SVM\testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n =np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSv, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('\n the training error rate is： %f' % ((float(errorCount))/float(m))*100)





"""""""""

手写字识别

"""""""""

'''
输入txt格式的图片，返回一行向量
'''
def img2vector(filename):
    returnVec = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,32*i+j] = int(lineStr[j])
    return returnVec


'''
载入数据，得到数据集和标签
'''
def loadImage(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


'''
手写字识别结果
'''
def testDigits(kTup=('rbf, 10')):
    dataArr,labelArr = loadImage(r'F:\Python\SVM\trainingDigits')
    b, alphas = smoP(dataArr,labelArr, 200, 0.0001, 10000, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs = datMat[svInd]
    labelSv = labelMat[svInd]
    print('there are %d Support Vectors' % np.shape(sVs)[0])
    m, n =np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], kTup)
        predict = kernelEval.T * np.multiply(labelSv, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('\n the training error rate is： %f' % ((float(errorCount)/m))*100)
    
    dataArr,labelArr = loadImage(r'F:\Python\SVM\testDigits')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n =np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], kTup)
        predict = kernelEval.T * np.multiply(labelSv, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('\n the training error rate is： %f' % ((float(errorCount))/float(m))*100)



if __name__ == '__main__': 
    
    print('-------SMO简单实现-------')
    dataArr,labelArr = loadDataSet(r'F:\Python\SVM\testSet.txt')
    print(dataArr)
    b, alphas = smoSimple(dataArr,labelArr, 0.6, 0.001, 40)
    print(b)
    
    print("-------SMO完整实现-------")
    b, alphas = smoP(dataArr,labelArr, 0.6, 0.001, 40)
    ws = calcWs(alphas, dataArr,labelArr)
    print(ws)
    dataMat = np.mat(dataArr)
    print(dataMat[0]*mat(ws)+b)
    print(labelArr[0])
    
    testRbf()
    
    print('-------手写识别效果-------')
    testDigits(('rbf', 20))
    
    
    
    
    
    
    