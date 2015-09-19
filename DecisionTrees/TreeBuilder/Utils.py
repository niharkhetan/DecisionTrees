'''
Created on Sep 17, 2015

@author: NiharKhetan
'''

import numpy as np
from DataParser.ReadCSV import *
from Bean.Feature import Feature

def getEntropy(feature):
    '''
    Gets the entropy for a given feature. 
    @param feature: type(object)
    @return entropy: type(float)
    '''
    n = feature.getCount()
    fSet = feature.getDiscreteSet()
    keys = fSet.keys()
    values = fSet.values()
    entropy = 0

    for v in values:
        entropy += (v / float(n)) * (np.log2(v / float(n)))
    return entropy*(-1) 

def getInformationGain(feature, classLabelFeature):
    '''
    calculates the information gain for a given feature
    @param feature: type(object) : feature of which you want to calculate information gain of
    @param classLabelFeature: type(object) : class label as given in the dataset.
    @return informationGain: type(float)
    '''        
    fSetFeature = feature.getDiscreteSet()    
    subLabels = []
    for eachKey in fSetFeature:
        indexOfEachFeature = []
        for i in range(0, feature.getCount()):
            if eachKey == feature.getData()[i]:
                indexOfEachFeature.append(i)
        data = []
        for eachIndex in indexOfEachFeature:
            data.append(classLabelFeature.getData()[eachIndex])
            
        subLabels.append(Feature("subF", data))
    
    sumSv = 0
    for subLabel in subLabels:
        sumSv -= subLabel.getCount()/float(classLabelFeature.getCount())*getEntropy(subLabel)
    informationGain = getEntropy(classLabelFeature) + sumSv
    return informationGain

def convertVectorToColumnar(vector):
    '''
    This returns the whole data as a list of features objects
    currently data is in form [[],[],[]]
    It converts it to [f0,f1,f2] where each fi is and object
    @param vector: type(list) vector which is the whole dataset
    @return columns: type(list) which is the whole data stored as columns
    '''
    columns = []
    columnLabels = vector[0]
    for column in range(0, len(columnLabels)):
        columnData = []
        for row in vector[1:]:
            columnData.append(row[column])
        columns.append(Feature(columnLabels[column], columnData))
    return columns

if __name__ == '__main__':

    training_data = "zoo-train.csv"
    vector = readFileAsVector(training_data)
    columns = convertVectorToColumnar(vector)
    classLabel = columns[16]
    print getEntropy(columns[16])
    
    print getInformationGain(columns[0], columns[16])
    infogainAllFeatures = []
    for column in columns[:-1]:
        infogainAllFeatures.append(getInformationGain(column, columns[-1]))
        
    print infogainAllFeatures
    print max(infogainAllFeatures)
    print infogainAllFeatures.index(max(infogainAllFeatures))