# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:20:18 2017

@author: yang
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""""""""

K-均值聚类

"""""""""
'''
导入数据
'''
def loadData(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


'''
计算欧式距离
'''
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB, 2)))


'''
随机生成质心
'''
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangJ * np.random.rand(k,1)
    return centroids


'''
K-均值聚类
'''
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distEclud(dataSet[i, :], centroids[j, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, 0)
    return centroids, clusterAssment
        
        
        
        
        
"""""""""

二分K-均值聚类

"""""""""        
def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, 0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(dataSet[j, :], np.mat(centroid0)) ** 2
        while(len(centList) < k):
            lowestSSE = np.inf
            for i in range(len(centList)):
                ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
                centroidMat, splitClusterAss = kMeans(ptsInCurrCluster, k, distMeas)
                sseSplit = np.sum(splitClusterAss[:, 1])
                seeNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i), 1])
                
                print('sseSplit, and notsplit:', sseSplit, seeNotSplit)
                if (sseSplit + seeNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCent = centroidMat
                    bestClustAss = splitClusterAss.copy()
                    lowestSSE = sseSplit + seeNotSplit
            
            bestClustAss[np.nonzero(bestClustAss[:, 0].A==1)[0], 0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:, 0].A==0)[0], 0] = bestCentToSplit
            print('最好切分点为：',bestCentToSplit)
            print('数据个数：', len(bestClustAss))
            centList[bestCentToSplit] = bestNewCent[0, :]
            centList.append(bestNewCent[1, :])
            clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment





"""""""""

Yahoo

"""""""""
'''
计算球面距离
'''
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = np.sin(vecA[0,1]*np.pi/180) * np.sin(vecB[0,1]*np.pi/180)
    b = np.cos(vecA[0,1]*np.pi/180) * np.cos(vecB[0,1]*np.pi/180) * np.cos(np.pi * (vecB[0,0]-vecA[0,0]) /180)
    return np.arccos(a + b)*6371.0


'''
绘制散点图
'''
def clusterClubs(numClust=4):
    datList = []
    for line in open(r'F:\Python\kMean\places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)    
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    print(myCentroids)
    print(np.shape(myCentroids))
    myCentroids = np.mat(np.reshape(myCentroids, [numClust, 2]))
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()









if __name__ == '__main__':
    print('-------k-均值聚类-------')
    dataMat = np.mat(loadData(r'F:\Python\kMean\testSet.txt'))
    myCent, myCluster = kMeans(dataMat, 4)
    
    print('-------二分K-均值聚类-------')
    dataMat3 = np.mat(loadData(r'F:\Python\kMean\testSet2.txt'))
    centList, clusterAssment = biKmeans(dataMat3, 3)

    print('-------Yahoo!-------')
    clusterClubs()






