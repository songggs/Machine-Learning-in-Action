# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:30:39 2017

@author: yang
"""
import sys
from numpy import mat, mean, power

'''
mapper
'''
def read_input(file):
    for line in file:
        yield line.rstrip()
        
inputt = read_input(sys.stdin)
inputt = [float(line) for line in inputt]
numInput = len(inputt)
inputt = mat(inputt)

sqInput = power(inputt, 2)
print('%d \t%f \t%f' % (numInput, mean(inputt), mean(sqInput)))
#print (sys.stderr, 'report: still alive')








