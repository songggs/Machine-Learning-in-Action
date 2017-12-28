# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:04:56 2017

@author: yang
"""

from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt



"""""""""

构造最简单的KNN

"""""""""

'''
创建数据集
return:
    group：数据的特征
    labels：数据的标签   
'''
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

    
'''
定义分类器,输入向量，根据给定数据集，判断其分类
para:
    inX：用于分类的输入向量,
    dataSet：输入的训练样本集,
    labels：输入的训练样本标签, 
    k：KNN的参数
return:
    输入向量的类别
'''
def classify0(inX, dataSet, labels, k):
    
    #计算给定点与数据集之间的距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    
    #距离排序，并返回索引
    sortedDistIndicies = distances.argsort()
    
    classCount = {}
    for i in range(k):
        #取得前k个标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #得到各个标签的数目
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    #返回类别
    return sortedClassCount[0][0]
    



"""""""""

改进网站的配对效果

"""""""""

'''
将txt文件转化为Numpy
para:
    filename：txt文件名
return：
    returnMat：数据特征 
    classLabelVector：数据标签
'''
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]

                
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

    
'''
将标签非数字的数据集转换成数据+标签
'''
def change_file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    label = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        line = line.split('\t')
        returnMat[index,:] = line[0:3]
        
        if line[-1] == 'largeDoses':
            label.append(1)
        elif line[-1] == 'smallDoses':
            label.append(2)
        else:
            label.append(3)
               
        index += 1
        
    return returnMat ,label


'''
画出数据两个特征的点图，并用不同的格式标出不同的标签
'''
def display():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(date[:,1], date[:,2], 15.0*array(label), 15.0*array(label))
    plt.show()

    
'''
归一化数据
para:
    dataSet：输入的数据集特征
return:
    normDataSet:归一化之后的特征, 
    ranges：每列特征的取值范围 
    minVals：每列的最小值   
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


'''
测试代码，输出错误率
'''
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = change_file2matrix('datingTestSet.txt')
    
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print("预测结果：%d，真正结果：%d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]): errorCount += 1
    print("错误率：%f" % (errorCount/float(numTestVecs)*100))


'''
#预测函数
根据输入的各个特征值，给出预测结果
'''
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTata = float(input('玩游戏的时间？'))
    ffMiles = float(input('每年飞的英里数？'))
    iceCream = float(input('每年吃的冰淇淋数？'))
    dataMat, Labels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(dataMat)
    inArr = array([ffMiles, percentTata, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, Labels, 3)

    print('你是否会喜欢他：', resultList[classifierResult - 1])
    



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
手写字识别，输出错误率
'''
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)        
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileName = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileName)
    for i in range(mTest):
        fileNameStr = testFileName[i]
        fileName = fileNameStr.split('.')[0]
        classNumStr = int(fileName.split('_')[0])

        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("预测结果为：%d，实际结果为：%d" % (classifierResult, classNumStr))

        if(classifierResult != classNumStr): errorCount += 1

    print("错误的个数为：%d" % errorCount)
    print("错误率为：%f" % (errorCount/float(mTest)))
        

    
if __name__ == '__main__':
    
    print("-------KNN简单实现-------")        
    group,labels = createDataSet()
    print(group,'\n',labels)
    answer = classify0([0,0],group,labels,3)
    print(answer)
    
    print("-------网站配对实现-------")
    date,label = file2matrix('datingTestSet2.txt')
    display()

    normMat, ranges, minVals = autoNorm(date)
    print(normMat)
    datingClassTest()
    classifyPerson()
    
    print("--------数字识别-------")
    handwritingClassTest()





    



















