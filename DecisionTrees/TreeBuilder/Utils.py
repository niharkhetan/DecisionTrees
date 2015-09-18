'''
Created on Sep 17, 2015

@author: NiharKhetan
'''

import numpy as np
from DataParser.ReadCSV import *
from Bean.Feature import Feature


def getEntropy(feature):
    n = feature.getCount()
    set = feature.getDiscreteSet()
    keys = set.keys()
    values = set.values()
    entropy = 0

    for v in values:
        entropy += (v / float(n)) * (np.log2(v / float(n)))
    return entropy*(-1) 

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

def convertVectorToColumnar(vector):
    '''This returns the whole data as a list of features objects'''
    columns = []
    columnLabels = vector[0]
    for column in range(0, len(columnLabels)):
        columnData = []
        for row in vector[1:]:
            columnData.append(row[column])
        columns.append(Feature(columnLabels[column], columnData))
    return columns

if __name__ == '__main__':
    '''
    positive = 5
    negative = 9
    entropy = getEntropy(positive, negative)
    print entropy
    
    print getInformationGain(entropy, [(14, 3, 4),(14, 6, 1)])
    print getInformationGain(entropy, [(14, 6, 2),(14, 3, 3)])
    '''
    training_data = "../DataParser/zoo-train.csv"
    vector = readFileAsVector(training_data)
    columns = convertVectorToColumnar(vector)
    classLabel = columns[16]
    print getEntropy(columns[16])