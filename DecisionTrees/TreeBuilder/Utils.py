'''
Created on Sep 20, 2015

@author   : NiharKhetan, Ghanshyam Malu
@desc     : Methods to calculate Entropy, Information Gain, Convert the data set vector to columnar and Vector Split to create dataset subsets

'''
from __future__ import division
import numpy as np
from DataParser.ReadCSV import readFileAsVector
from Bean.Feature import Feature
from Bean.Node import *

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

def vectorSplit(columnnarFeatureVector, classLabelFeature, splitFeature):
    '''
    Split the vector on a given SplitFeature and return a collection of sub-vectors 
    @param columnnarFeatureVector: type(list) Feature Vector dataset in the columnar fashion 
    @return newSubFeatureCollection: type(list) Collection of sub-vectors which exclude the splitFeature
    '''
    
    # Collection to hold new sub vector datasets        
    newSubFeatureCollection = []
       
    # For every discrete value in the split feature, create a new sub-feature list excluding the split feature
    for discreteVal in splitFeature.getDiscreteSet().keys():
        
        subFeatureList = []
        
        # Get the indices of all the data points in the splitFeature which match the current discrete value
        matchDataIndexList = [i for i,x in enumerate(splitFeature.getData()) if x == discreteVal]
        
        # Get the data points of every feature from the original vector dataset which match the index found above
        for i in range(len(columnnarFeatureVector)):
            
            # Exclude the data points of split feature
            if columnnarFeatureVector[i] != splitFeature:
                
                # Get all the data points of the current feature which match the index of the current discrete value of the splitFeature
                tempFeatureData = [ columnnarFeatureVector[i].getData()[idx] for idx in matchDataIndexList]

                # Create a new feature with the filtered data AND append it the feature list of the current discrete value
                subFeatureList.append(Feature(columnnarFeatureVector[i].getName(), tempFeatureData))
    
        # Append the final sub-feature list for the current discrete value into the master sub-feature collection 
        newSubFeatureCollection.append([str(discreteVal), subFeatureList])
        #newSubFeatureCollection.append(subFeatureList)
    
    return sorted(newSubFeatureCollection)

if __name__ == '__main__':
    pass