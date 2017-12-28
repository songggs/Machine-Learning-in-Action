# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:31:26 2017

@author: yang
"""
from numpy import *
import re
import sys

#sys.path.append("E:/")

""""""""""

基于文本分类的朴素贝叶斯的简单实现

"""""""""

'''
构造数据
得到文档数据和文档标签
'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec
    
 
'''
 #词表到向量的转换 
给定一个文本数据集，得到一个单词集合
给定单词集合和待测试的文档，判断文档中是否存在单词集合中的单词，得到0&1集合
'''  
def createVocabList(dataSet):
    vocabSet = set([])                      #创建一个空的不重复列表
    for document in dataSet:               
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

def setOfWords2Vec(vocabSet, inputSet):
    returnVec = [0] * len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            returnVec[vocabSet.index(word)] = 1
        else: 
            print("%s 这个单词不在集合中！" % word)
    return returnVec


'''
#训练函数
para:
    trainMatrix：训练集文档, 
    trainCategory：训练集标签，即文档是否是侮辱性文档
return:
    P0Vect：p(w|c0), 
    P1Vect：p(w|c1), 
    pAbusive：p(c1)
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            
    P1Vect = log(p1Num/p1Denom)
    P0Vect = log(p0Num/p0Denom)
    return P0Vect, P1Vect, pAbusive


'''
分类函数,输入待测试数据，得到分类结果
para:
    vec2Classify为待测试的数据, 
    p0Vec为p(w/c0),
    p1Vec为p(w/c1), 
    pClass1为为p(c1)
'''    
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOposts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOposts)
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classify as:', classifyNB(thisDoc, p0V, p1V, pAb))

'''
词袋模型，给定单词集合和输入文档，得到各个单词出现的次数的列表
'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
    
 
    
    
"""""""""

过滤垃圾邮件

"""""""""
'''
对邮件文本进行切分
'''
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]


'''
测试
'''

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open(r'spam\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(r'ham\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
        
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pAb = trainNB0(array(trainMat), array(trainClasses))
        
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pAb) != classList[docIndex]:
            errorCount += 1
    print('the error rate is：', float(errorCount)/len(testSet))


        
        

if __name__ == '__main__':
    
    print('-------基于文本分类的朴素贝叶斯的简单实现-------')
    listOposts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOposts)
    print(myVocabList)
    answer = setOfWords2Vec(myVocabList, listOposts[0])
    print(answer)
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
         
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print(p0V, p1V, pAb)
    
    testingNB()    
    
    print('-------邮件过滤-------')
    spamTest() 
    
    
    
   