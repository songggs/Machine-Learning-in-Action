# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:52:07 2017

@author: yang
"""

"""""""""

构建FP-树

"""""""""

'''
创建FP树的数据结构
'''
class treeNode:
    #初始化
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
        
    #对count变量增加给定值
    def inc(self, numOccur):
        self.count += numOccur
        
    #将树以文本形式显示
    def disp(self, ind = 1):
        print('  '*ind, self.name, '  ', self.count)
        for child in self.children.values():
            child.disp(ind+1)


'''
创建树
'''
def createTree(dataSet, minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
            if len(localD)>0:
                orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
                updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


'''
更新树
'''
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items)>1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        

'''
更新头指针表，确保节点链接指向树中的每个实例
'''
def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


'''
简单数据集及数据包装器
'''
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x','w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q','t', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


'''
抽取条件模式基
'''
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
        
        
def findPrefixPath(basePat, treeNode):   
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats
        
'''        
生成条件树
'''     
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1])]
    
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCond, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            mineTree(myCond, myHead, minSup, newFreqSet, freqItemList)

     



if __name__ == '__main__':
    tree = treeNode('yang', 23, 'sha')
    tree.children['nuan'] = treeNode('nuan', -5, 'Mr')
    tree.disp()
    Data = loadSimpDat()
    D = createInitSet(Data)

    
    tree, head = createTree(D, 1)
    tree.disp()
    
    cond = findPrefixPath('x', head['x'][1])
    print(cond)
    
    freqItems=[]
    mineTree(tree, head, 1, set([]), freqItems)
    
