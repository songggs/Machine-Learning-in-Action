# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:03:32 2017

@author: yang
"""

"""""""""

Apriori算法

"""""""""
def loadDataSet():
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]


"""
生成频繁项集
"""

'''
生成C1
'''
def createC1(dataSet):
    C1= []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


'''
从Ck生成Lk
para：
    D：数据集, 
    Ck：指的是第几个C, 
    minSupport：最小支持度
return:
    retList：Lk, 
    supportData：包含支持度的字典
'''
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can in ssCnt:
                    ssCnt[can] += 1                    
                else:
                    ssCnt[can] = 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


'''
由Lk-1创建Ck
para:
    Lk：上一次生成的L，即频繁项集
    k：频繁项集中，每个子集的的长度
'''
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


'''
apriori
'''
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


"""
生成关联规则
"""

'''
对生成的规则进行评估，是否满足最小置信度
para:
    freqSet:每个频繁项集的子集, 
    H：freqSet中的每个元素或几个元素组成的集合, 
    supportData：频繁项集的数据字典, 
    br1：待填充的列表，即最后返回包含可信度的规则列表, 
    minConf：最小置信度阈值
return：
    满足最小置信度的列表   
'''
def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '--->', conseq, 'conf:', conf)
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


'''
生成候选规则集合
'''
def rulesFromConseq(freqSet, H, supportData, br1, minConf = 0.7):
    m = len(H[0])
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H, m+1)
        calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


'''
生成包含可信度的规则集合
para:
    L：频繁项集合, 
    supportData：频繁项集合数据字典
    
'''
def generateRules(L, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            #如果freqSet中的元素超过两个，需要进行规则细分
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            #如果freqSet中的元素等于两个，直接轮流作为右面元素即可
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList





if __name__ =='__main__':
    print('-------生成频繁项集-------')
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet)
    print(L)

    print('-------生成关联规则-------')
    rules = generateRules(L, supportData, minConf = 0.7)
    
    print('-------毒蘑菇-------')
    mushDatSet = [line.split() for line in open(r'F:\Python\Apriori\mushroom.dat').readlines()]
    L, suppData = apriori(mushDatSet, minSupport = 0.3)
    print(L)
    
    
    
    
    
    

