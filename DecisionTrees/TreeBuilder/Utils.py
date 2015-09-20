'''
Created on Sep 17, 2015

@author: NiharKhetan
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

def buildDecisionTree(columnarFeatureVector, node, discreteValue, depth):
    ''' 
    Build decision tree 
    '''
    # Global declaration
    global decisionTreeDict, depthLimit

    # Stop if max depth reached
    if depth == depthLimit or len(columnarFeatureVector)==1:
        # Get the Majority / Most repeated class label of the training data set to assign the classification label to the current attribute
        majorityClassLabel = columnarFeatureVector[-1].getDiscreteSet().most_common(1)[0][0]
        # Build Leaf Node and add it as a child to the current node
        newLeafNode = Node(majorityClassLabel, discreteValue ,None, [], depth, True)
        node.addChildNode(newLeafNode)
        if decisionTreeDict.get(node.getNode()) is None:
            decisionTreeDict[node.getNode()] = {}
        decisionTreeDict[node.getNode()][discreteValue] = majorityClassLabel
        return node
            
    # Find entropy of the class label
    entropy = getEntropy(columnarFeatureVector[-1])
    
    # If entropy zero, leaf reached. Terminate 
    if entropy == 0:
        # Build Leaf Node and add it as a child to the current node
        newLeafNode = Node(columnarFeatureVector[-1].getData()[0], discreteValue ,None, [], depth, True)
        node.addChildNode(newLeafNode)
        if decisionTreeDict.get(node.getNode()) is None:
            decisionTreeDict[node.getNode()] = {}
        decisionTreeDict[node.getNode()][discreteValue] = columnarFeatureVector[-1].getData()[0]
        return node
    
    # Find information gain of all the features
    infogainAllFeatures = []
    for column in columnarFeatureVector[:-1]:
        infogainAllFeatures.append(getInformationGain(column, columnarFeatureVector[-1]))
            
    # Index of the feature with maximum info gain
    splitNodeIndex = infogainAllFeatures.index(max(infogainAllFeatures))
    
    print "\n\n","*" * 80
    if node is None:
        # Build tree for the node: root
        print ('Build Tree \tDepth : %s \tNode: root' % (depth))
        newDecisionNode = Node(columnarFeatureVector[splitNodeIndex].getName(), 'root', columnarFeatureVector[splitNodeIndex], [], depth)
        node = newDecisionNode
    
    else:        
        print ('Build Tree \tDepth : %s \tNode: %s \tAttribute: %s' % (depth, node.getNode(), node.getAttribute()))
        newDecisionNode = Node(columnarFeatureVector[splitNodeIndex].getName(), discreteValue, columnarFeatureVector[splitNodeIndex], [], depth)
        node.addChildNode(newDecisionNode)
    print "*" * 80,"\n"

    # Decision Tree Dictionary to maintain the "attribute" as key and "look up feature" as values 
    if decisionTreeDict.get(node.getNode()) is None:
        decisionTreeDict[node.getNode()] = {}
    
    # Add "attribute" as key for nodes other than Root
    if discreteValue is not None:
        decisionTreeDict[node.getNode()][discreteValue] = newDecisionNode.getNode()

    # Split the vector feature data set
    newSubFeatureCollection = vectorSplit(columnarFeatureVector, columnarFeatureVector[-1], columnarFeatureVector[splitNodeIndex])
    
    print "Number of Children:", len(newSubFeatureCollection)
    print "Splitting on :", columnarFeatureVector[splitNodeIndex].getName(),"\n"
    
    # Print Node details
    for subFeatureBucket in newSubFeatureCollection:
        print "DiscreteVal:", subFeatureBucket[0], "\t| ",
        for feature in subFeatureBucket[1]:
            padding = " |"
            if feature.getCount() > 9:
                padding = "|"
            print feature.getName(), "-", feature.getCount(), padding,
        print

    # Recurse over each subFeature in the new feature collection for each discrete value of the split feature
    for subFeatureBucket in newSubFeatureCollection:
        buildDecisionTree(subFeatureBucket[1], newDecisionNode, subFeatureBucket[0], depth+1 )   
              
    return node

def printTree(node):
    '''
    Prints the Decision Tree using recursion
    ''' 
    # Return if node is leaf
    if node.isLeaf():
        return
    
    # Print attributes of node
    print "\n","-"*90
    print "Attribute:",node.getAttribute(),
    print "\tNode:",node.getNode(), 
    print "\tLevel:", node.getLevel(),
    print "\tChildren Count:", len(node.getChildren())
    print "\n","-"*90
    
    # Print attributes of children 
    children = node.getChildren()
    if children is not None and len(children) > 0:
        print "\t***Children***"
        for i in range(len(children)):
            print "\tAttribute:",children[i].getAttribute(),
            print "\tNode:",children[i].getNode(),
            print "\tLevel:",  children[i].getLevel(),
            if children[i].isLeaf():
                print "\t ClassificationLabel:",children[i].getNode(), "\t****** Leaf *****",
            print
            
    # Recurse to print each children
    if children is not None and len(children) > 0:
        for i in range(len(children)):
            printTree(children[i])

def trainModel(training_data):
    ''' Train the model '''
   
    # Global declaration
    global rootDecisionTree
    
    # Read the CSV data set file as vector
    vector = readFileAsVector(training_data)
    
    # Convert the data set into columnar format
    columnarVectorDataset = convertVectorToColumnar(vector)
    
    # Build decision tree
    rootDecisionTree = buildDecisionTree(columnarVectorDataset, None, None, 0)
    
    return rootDecisionTree
        
def classifyDataPoint(dataPoint, featureNameList, majorityClassLabel):
    ''' Classifies the given data point to one of the Class Labels''' 
    # Global declaration
    global rootDecisionTree,decisionTreeDict
    
    # Get the initial feature to check 
    featureToCheck = rootDecisionTree.getNode()
    
    # Initialize classificationLabel
    classificationLabel = None
    
    while featureToCheck in featureNameList:  
        # Get the data point value for the feature
        attribute = dataPoint.get(featureToCheck)
        
        # Get the 'possible attributes' dictionary for the feature
        attributeDict = decisionTreeDict.get(featureToCheck)
        
        # Fetch the next feature to check
        featureToCheck = attributeDict.get(attribute)

        # Classification Label found
        if featureToCheck not in featureNameList:
            classificationLabel = featureToCheck
            
    # Unseen test data, assign to the majority Class Label
    if classificationLabel is None:
        classificationLabel = majorityClassLabel

    return classificationLabel

def computeConfusionMatrix(actualClassLabelList, predictedClassLabelList):
    
    # Global declaration
    global depthLimit
    
    print "\n","="*90
    print "\t\t\t\tCompute Confustion Matrix"
    print "="*90

    print actualClassLabelList
    print predictedClassLabelList
    
    totalDatasetCount = len(actualClassLabelList)
    correctClassificationCount = 0
    incorrectClassificationCount = 0
    
    for i in range(len(actualClassLabelList)):
        if actualClassLabelList[i] != predictedClassLabelList[i]:
            incorrectClassificationCount += 1
    
    print "\nIncorrect Classification Count: %d \tTotal Dataset Count: %d" %(incorrectClassificationCount, totalDatasetCount)
    
    errorRate = incorrectClassificationCount / totalDatasetCount
    
    print "\n","*"*90
    print "\t!!!!!!!\t\tError Rate at DepthLimit (%d) is %f\t!!!!!!!" % (depthLimit, errorRate)
    print "*"*90
    
    
def testModel(test_data):
    ''' Test the model '''

    # Global declaration
    global rootDecisionTree,decisionTreeDict

    # Create vector of the test data set
    vectorDataSet = readFileAsVector(test_data)
    
    # Convert to columnar to get all the classification labels easily
    columnarVectorDataset = convertVectorToColumnar(vectorDataSet)
    actualClassLabelFeature = columnarVectorDataset[-1]
    actualClassLabelList = actualClassLabelFeature.getData()
    
    # Get the Majority / Most repeated class label of the training data set to handle unseen data points
    majorityClassLabel = actualClassLabelFeature.getDiscreteSet().most_common(1)[0][0]
    
    # Predicted Class Label list of the test data
    predictedClassLabelList = []
    
    # Get the feature names 
    featureNameList = vectorDataSet[0]
    
    # Find the classification label for every data point
    for dataPoint in vectorDataSet[1:] :
        
        # Dictionary to maintain the values of all the features of the current data point
        tempDataPointDict = {}
        for i in range(len(featureNameList)):
            tempDataPointDict[featureNameList[i]] = dataPoint[i]
        
        # Classify the data point
        currentPredictedClassLabel = classifyDataPoint(tempDataPointDict, featureNameList, majorityClassLabel)
        
        # Append to the main Predicted Class Label list
        predictedClassLabelList.append(currentPredictedClassLabel)
        
    return actualClassLabelList, predictedClassLabelList   
    
def printDecisionTree():
    ''' Print Decision Tree with formatting'''
    # Global declaration
    global rootDecisionTree
    
    print "\n","*"*90
    print "\t\t\t\t Decision Tree"
    print "*"*90
    printTree(rootDecisionTree)
        
def printDecisionTreeDict():
    ''' Print Decision Tree Dictionary with formatting'''
    # Global declaration
    global decisionTreeDict
    print "\n","*"*90
    print "\t\t\t\t Decision Tree Dictionary"
    print "*"*90    
    for key, value in decisionTreeDict.iteritems():
        print key,":",value
         
def main():
    ''' Main '''
    # Global declaration
    global rootDecisionTree,decisionTreeDict
    
    training_data = "zoo-train_withoutFirstFeature.csv"
    test_data = "zoo-test_withoutFirstFeature.csv"
    
    training_data = "carvana_train.csv"
    test_data = "carvana_test.csv"
    
    # Train model
    trainModel(training_data)
    
    # Print decision tree
    # printDecisionTree()
    
    # Print decision tree dict
    # printDecisionTreeDict()
    
    # Test model
    actualClassLabelList, predictedClassLabelList = testModel(test_data)
    
    computeConfusionMatrix(actualClassLabelList, predictedClassLabelList)
    
if __name__ == '__main__':
    
    # Global variables
    rootDecisionTree = None
    decisionTreeDict = {}   #Master dictionary
    depthLimit = 5
    main()
    