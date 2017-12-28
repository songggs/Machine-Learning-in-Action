# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:08:41 2017

@author: yang
"""

import sys
from numpy import mat, mean, power

'''
reduce
'''

def read_input(file):
    for line in file:
        yield line.rstrip()
        
inputt = read_input(sys.stdin)
mapperOut = [line.split('\t') for line in inputt]
cumVal = 0.0
cumSymSq = 0.0
cumN = 0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj * float(instance[1])
    cumSymSq += nj * float(instance[2])
mean = cumVal/cumN
varSum = (cumSymSq - 2*mean*cumVal + cumN*mean*mean)/cumN
print(' %d \t %f \t %f' % (cumN, mean, varSum))
#print (sys.stderr, 'report: still alive')
















