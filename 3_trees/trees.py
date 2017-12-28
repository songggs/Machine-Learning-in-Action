# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:27:13 2017

@author: yang
"""

from math import log

"""""""""

决策树实现

"""""""""

'''
#定义一个简单的数据集
return:
    dataSet：带标签的数据集，最后一列为标签, 
    labels：各个特征的名称
'''
def createDataSet():
    dataSet = [[1, 1, 'yes'], 
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [1, 0, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


'''
#计算给定数据集的香农熵
给定数据集，返回数据集的总熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for fect in dataSet:
        currentLabel = fect[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


'''
#按照给定特征划分数据集，返回划分完的数据集
para:
    dataSet：给定数据集, 
    axis：用来划分数据的特征的索引值, 
    value：用来划分数据的特征的其中一个取值
return:
    划分后，去掉划分特征的数据集
'''
def splitDataSet(dataSet, axis, value):
    retData = []
    for fect in dataSet:
        if fect[axis] == value:
            reduceFect = fect[:axis]
            reduceFect.extend(fect[axis+1:])
            retData.append(reduceFect)
    return retData


'''
#选择最好的特征，返回最好的特征下标
给定数据集，按照每个特征进行划分数据，并计算各个的熵，返回最好的特征的索引值
'''    
def chooseBestFeatureToSplit(dataSet):
    #特征数
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueFeat = set(featList)
        newEntropy = 0.0
        
        for value in uniqueFeat:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


'''
#规定叶节点的类别
给定叶节点中各个数据的类别列表，取个数最多的类别为叶节点的类别
'''    
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


'''
#创建树
para:
    dataSet:包含标签的数据集, 
    labels：对应各个特征的名称    
'''
def creatTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    mytree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValue = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValue)
    for value in uniqueVals:
        subLabels = labels[:]
        mytree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return mytree

    
'''
测试决策树
para:
    inputTree：已训练好的决策树, 
    featLabels：各个特征对应的名称, 
    testVec：待测试的数据
return:
    输入数据预测的标签
'''
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]                                                        #获取决策树结点
    secondDict = inputTree[firstStr]                                                        #下一个字典
    featIndex = featLabels.index(firstStr)                                               
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel
    
    
'''
决策树的存储&重载
'''
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
    
    
    

"""""""""

绘制树形图

"""""""""
import matplotlib.pyplot as plt
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

'''
#这个是用来一注释形式绘制节点和箭头线
'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',xytext=centerPt,
                            textcoords='axes fraction', va="center", ha="center", bbox=nodeType,
                            arrowprops=arrow_args)


'''
获取叶节点数目和树的层数
'''
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNUmLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs
    
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict.keys()).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth>maxDepth:
            maxDepth = thisDepth
    return maxDepth
        
        
'''
给决策树的箭头添加文本
para:
    cntrPt:箭头的起点
    parentPt:箭头的终点, 
    txtString：待添加的文本
'''    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
    

def plotTree(myTree, parentPt, nodeTxt):
    leafs = getNUmLeafs(mytree)
    depth = getTreeDepth(mytree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(leafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + plotTree.totalW            
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


'''    
#真正的绘制
'''
def createPlot0():    
    fig = plt.figure(1, facecolor='white')
    # Clear figure
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('决策节点', (0.5,0.1), (0.1,0.5), leafNode)
    plt.show()


def createPlot(inTree):    
    fig = plt.figure(1, facecolor='white')
    # Clear figure
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()






if __name__ == '__main__':
    
    print('-------构造简单的决策树-------')
    data, label = createDataSet()
    shannon = calcShannonEnt(data)
    print(shannon)
    feat = chooseBestFeatureToSplit(data)
    print(feat) 
    mytree = creatTree(data, label) 
    print(mytree) 

    print('-------决策树的测试-------')
    inputTree = {'flippers': {0: 'no', 1: {'no surfacing': {0: 'no', 1: 'yes'}}}}
    label = ['flippers', 'no surfacing']
    answ = classify(inputTree, label, [1,1])
    print(answ)

    print('-------决策树保存&重载-------')
    storeTree(inputTree,'classifierStorage.txt')
    haha = grabTree('classifierStorage.txt')
    print(haha)

    print('-------绘制树形图-------')
    createPlot0()
    
    leafs = getNUmLeafs(mytree)
    depth = getTreeDepth(mytree)
    print(leafs, depth)

    createPlot(mytree)




  
    
    






