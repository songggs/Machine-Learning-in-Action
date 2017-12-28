# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:57:23 2017

@author: yang
"""

from numpy import *


def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
    
'''
三种相似度计算
'''
def euclidSim(inA, inB):
    return 1.0/(1.0 + linalg.norm(inA - inB))


def pearsSim(inA, inB):
    if(len(inA) < 3):
        return 1.0
    return 0.5 + 0.5 *corrcoef(inA, inB, rowvar = 0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5*(num/denom)


'''
估计用户对物体的评分
para:
    dataMat:数据矩阵, 
    user：用户编号, 
    simMeas：计算相似度的方法, 
    item：物品编号
return:物品的评分
'''
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:, j].A>0, dataMat[:, item].A>0))[0]
        if len(overLap)==0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, j], dataMat[overLap, item])
        print('物品%d 和物品 %d 的相似度为: %f' % (j, item, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal
    
    
'''
产生最高的N个推荐结果
para:
    dataMat:数据集, 
    user：用户编号, 
    N：推荐的个数, 
    simMeas：相似度度量方法, 
    estMethod：评分估计方法
return：评分最高的N个推荐结果
'''
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems)==0:
        return '已经全部评分'
    itemScore = []
    for item in unratedItems:
        estimatedSored = estMethod(dataMat, user, simMeas, item)
        itemScore.append((item, estimatedSored))
    ans = sorted(itemScore, key = lambda jj:jj[1], reverse=True)
    return ans[:N]

"""""""""

基于SVD的评分估计

"""""""""
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, sigma,VT = linalg.svd(dataMat)
    sig4 = mat(eye(4) * sigma[:4])
    xformedItems = dataMat.T * U[:, :4] *sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j==item:
            continue
        similarity = simMeas(xformedItems[j, :].T, xformedItems[item, :].T)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


"""""""""
图像压缩
"""""""""
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print (1, end = ' ')
            else:
                print(0, end = ' ')
        print(' ')
        

'''
实现图像的压缩
'''
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open(r'F:\Python\SVD\0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    mymat = mat(myl)
    print('*******原始图片*******')
    printMat(mymat, thresh)

    U, sigma, VT = linalg.svd(mymat)
    sigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        sigRecon[k,k] = sigma[k]
    recMat = U[:, :numSV] * sigRecon * VT[:numSV, :]
    print('********用 %d 个奇异值重构的矩阵*******' % numSV)
    printMat(recMat, thresh)












if __name__ == '__main__':
    data = loadExData()
    U, sigma, VT = linalg.svd(data)
    print('U:',U)
    print('sigma:',sigma)
    print('VT:',VT)
    print('数据重构')
    sig3 = mat([[sigma[0], 0, 0], [0, sigma[1], 0], [0, 0, sigma[2]]])
    conMat = U[:, :3] * sig3 * VT[:3, :]
    print(conMat)
    
    print('-------餐馆推荐-------')
    data = [[4,4,0,2,2],
            [4,0,0,3,3],
            [4,0,0,1,1],
            [1,1,1,2,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0]]
    data = mat(data)
    print(data)
    ans = recommend(data, 2)
    print(ans)
    
    print('---------使用SVD方法---------')
    mydata = loadExData2()
    U, sigma, VT = linalg.svd(mydata)
    print(sigma)
    print(sum(sigma**2)*0.9)
    print(sum(sigma[:3]**2))
    
    print('--------两种度量方式比较-------')
    mydata = mat(mydata)
    met1 = recommend(mydata, 1, 3, pearsSim, svdEst)
    print(met1)
    met2 = recommend(mydata, 1, 3, cosSim, svdEst)
    print(met2)
    
    
    imgCompress()
    
    
    