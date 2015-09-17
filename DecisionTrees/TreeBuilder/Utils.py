'''
Created on Sep 17, 2015

@author: NiharKhetan
'''

import numpy as np

def getEntropy(positive, negative):
    pos = positive / float(positive + negative)
    neg = negative / float(positive + negative)
    entropy = -1*(pos*np.log2(pos)) - (neg*np.log2(neg))
    return entropy

def getInformationGain(entropy, entropySv):
    '''entropySv is a list of tuple (x, y, z):
    x is total no of items in data set
    y is no of + items
    z is no of - items'''
    sumSv = 0
    for Sv in entropySv:
        sumSv -= (Sv[1]+Sv[2])/float(Sv[0])*getEntropy(Sv[1], Sv[2])
    informationGain = entropy + sumSv
    return informationGain

if __name__ == '__main__':
    positive = 5
    negative = 9
    entropy = getEntropy(positive, negative)
    print entropy
    
    print getInformationGain(entropy, [(14, 3, 4),(14, 6, 1)])
    print getInformationGain(entropy, [(14, 6, 2),(14, 3, 3)])