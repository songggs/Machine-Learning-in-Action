# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:47:07 2017

@author: yang
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

'''
载入数据
'''
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


'''
将数据切分成两个子集
para：
    dataSet：数据集
    feature:待切分特征
    value:特征值
return：两个子集
'''
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0],:]
    return mat0, mat1


'''
创建叶节点函数
返回数据集的均值，作为叶节点的标签
'''
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


'''
计算总方差，用于度量特征选取好坏
'''
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


'''
回归树的切分，得到最好特征的索引和特征值
para:
    dataSet:数据集, 
    leafType=regLeaf：创建叶节点函数，得到叶节点标签, 
    errTpye=regErr：计算总方差, 
    ops=(1, 4)：容许误差的下降值和切分的最小样本数
                当下降值小于ops[0]时，退出
                当切分样本数小于最小样本数时，退出
return:
    bestIndex：最好的特征
    bestValue：最好特征的值
'''
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n =np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for feat in range(n-1):
        for splitVal in set(dataSet[:, feat].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, feat, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = feat
                bestValue = splitVal
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


'''
创建树
para:
    dataSet:数据集
'''
def creatTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = creatTree(lSet, leafType, errType, ops)
    retTree['right'] = creatTree(rSet, leafType, errType, ops)
    return retTree





"""""""""

回归树的剪枝

"""""""""    
'''
判断目标是否是树,返回bool值
'''
def isTree(obj):
    return (type(obj).__name__ == 'dict')


'''
递归函数，从上向下遍历树，直到叶节点为止，返回两个叶节点的平均值
也称对树进行塌陷处理
'''
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0


'''
回归树剪枝
para：
    tree：待剪树
    testData：测试数据集
return：
    剪完之后的树
'''
def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['right']):
        prune(tree['right'], rSet)
    if isTree(tree['left']):
        prune(tree['left'], lSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(tree['left'] - lSet[:, -1], 2)) + np.sum(np.power(tree['right'] - rSet[:, -1], 2))
        treeMean = (tree['right'] + tree['left'])/2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree





"""""""""

模型树

"""""""""
'''
普通线性拟合
'''
def linerSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('矩阵为奇异矩阵，不能求逆')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


'''
模型树叶节点函数
'''
def modelLeaf(dataSet):
    ws, x, y = linerSolve(dataSet)
    return ws


'''
模型树误差
'''
def modelErr(dataSet):
    ws, x, y = linerSolve(dataSet)
    yHat = x * ws
    return np.sum(np.power(yHat-y, 2))


"""""""""

树回归预测

"""""""""
'''
回归树叶节点函数
'''
def regTreeEval(model, inDat):
    return float(model)


'''
模型树叶节点函数
'''
def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)


'''
单个数据，用树划分
'''
def treeForeCast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
    
    
'''
整个测试集，用树划分
'''
def creatForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat
    




"""""""""

使用tkinter创建GUI

"""""""""
from tkinter import *
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

'''
给定tolS和tolN，判断复选框是否选中，从而判断采用回归树还是模型树
并画出训练集的散点图和测试集的连续图
'''
def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        mytree = creatTree(reDraw.rawDat, modelLeaf, modelErr, (tolS, tolN))
        yHat = creatForeCast(mytree, reDraw.testDat, modelTreeEval)
    else:
        mytree = creatTree(reDraw.rawDat, ops = (tolS, tolN))
        yHat = creatForeCast(mytree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5, c='r')
    reDraw.a.plot(reDraw.testDat, yHat, linewidth = 2.0)
    reDraw.canvas.show()
        
        
'''
读取文本框中的数字
'''  
def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print('请输入整型的tolN')
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 0.1
        print('请输入浮点型的tolS')
        tolSentry.delete(0, END)
        tolSentry.insert(0, '0.1')
    return tolN, tolS


'''
画出树
'''      
def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)




    
    
    
    
    
if __name__ == '__main__':
    print('-------创建树-------')
    myDat = loadDataSet(r'F:\Python\regTrees\ex00.txt')
    myDat = np.mat(myDat)    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(myDat[:,0].flatten().A[0], myDat[:, 1].flatten().A[0], marker = 'o', color = 'b', s = 30)
    plt.show()        
    myTree = creatTree(myDat)
    print(myTree)
    
    myDat = loadDataSet(r'F:\Python\regTrees\ex0.txt')
    myDat = np.mat(myDat)    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(myDat[:,1].flatten().A[0], myDat[:, 2].flatten().A[0], marker = 'o', color = 'b', s = 30)  
    plt.show()      
    myTree = creatTree(myDat)
    print(myTree)
    
    print('-------树剪枝--------')
    myDat2 = loadDataSet(r'F:\Python\regTrees\ex2.txt')
    myDat2 = np.mat(myDat2)
    myTree2 = creatTree(myDat2, ops=(0,1))
    print(myTree2)
    
    myDataTest = loadDataSet(r'F:\Python\regTrees\ex2test.txt')
    myDataTest = np.mat(myDataTest)
    tree = prune(myTree2, myDataTest)
    print(tree)

    myData = np.mat(loadDataSet(r'F:\Python\regTrees\exp2.txt'))
    tree = creatTree(myData, modelLeaf, modelErr, (1, 10))
    print(tree)

    print('-------树回归与标准回归的比较-------')
    trainMat = np.mat(loadDataSet(r'F:\Python\regTrees\bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet(r'F:\Python\regTrees\bikeSpeedVsIq_test.txt'))
    myTree = creatTree(trainMat, ops=(1, 20))
    yHat = creatForeCast(myTree, testMat[:, 0])
    anw = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print('回归树：',anw)
    
    myTree = creatTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat = creatForeCast(myTree, testMat[:, 0], modelTreeEval)
    anw1 = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print('模型树：',anw1)
    
    ws, X, Y = linerSolve(trainMat)
    for i in range(np.shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    anw2 = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print('标准回归:', anw2)

    
    
    
    root = Tk()
    
    '''
    tk上一些小框架初始化
    '''
    Label(root, text = "Plot Place Holder").grid(row=0, columnspan=3)
    Label(root, text = "tolN").grid(row=1, column=0)
    tolNentry = Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    
    Label(root, text = 'tolS').grid(row=2,column=0)
    tolSentry = Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')
    
    Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)
    
    chkBtnVar = IntVar()
    chkBtn = Checkbutton(root, text='Model tree', variable=chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)
    
    #按钮初始化
    reDraw.rawDat = np.mat(loadDataSet(r'F:\Python\regTrees\sine.txt'))
    reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
    
    reDraw.f = Figure(figsize=(5,4), dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
    
    
    drawNewTree()
    
    
    
    root.mainloop()
