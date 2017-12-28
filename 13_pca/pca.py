# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:53:06 2017

@author: yang
"""

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

'''
加载数据集
'''
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [list(map(float, line)) for line in stringArr]
    return mat(dataArr)


'''
pca
'''
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


"""""""""
PCA对半导体数据降维
"""""""""
def replaceNanWithMean():
    datMat = loadDataSet(r'F:\Python\pca\secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat
    







if __name__ == '__main__':
    data = loadDataSet(r'F:\Python\pca\testSet.txt')
    lowMat, recMat = pca(data, 1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0].flatten().A[0], data[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(recMat[:, 0].flatten().A[0], recMat[:, 1].flatten().A[0], marker='o', c='r')


    print('半导体')
    bdata = replaceNanWithMean()
    meanVals = mean(bdata, axis=0)
    meanRemoved = bdata - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    
    ans = eigVals[0]/sum(eigVals)
    print(ans)










