'''
Created on Sep 17, 2015

@author: NiharKhetan
'''
from Utils import *
import sys

def buildDecisionTree(columnarFeatureVector, node, discreteValue, depth):
    ''' 
    Build decision tree 
    '''
    # Global declaration
    global decisionTreeDict, depthLimit, printDecisionTreeBuildProcessFlag

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
    
    sys.stdout.write ( "\n\n" + "*" * 80 + "\n") if printDecisionTreeBuildProcessFlag == True else None
    if node is None:
        # Build tree for the node: root
        sys.stdout.write ('Build Tree \tDepth : %s \tNode: root\n' % (depth)) if printDecisionTreeBuildProcessFlag == True else None
        newDecisionNode = Node(columnarFeatureVector[splitNodeIndex].getName(), 'root', columnarFeatureVector[splitNodeIndex], [], depth)
        node = newDecisionNode
    
    else:        
        sys.stdout.write('Build Tree \tDepth : %s \tNode: %s \tAttribute: %s\n' % (depth, node.getNode(), node.getAttribute())) if printDecisionTreeBuildProcessFlag == True else None
        newDecisionNode = Node(columnarFeatureVector[splitNodeIndex].getName(), discreteValue, columnarFeatureVector[splitNodeIndex], [], depth)
        node.addChildNode(newDecisionNode)
    sys.stdout.write("*" * 80 + "\n") if printDecisionTreeBuildProcessFlag == True else ''

    # Decision Tree Dictionary to maintain the "attribute" as key and "look up feature" as values 
    if decisionTreeDict.get(node.getNode()) is None:
        decisionTreeDict[node.getNode()] = {}
    
    # Add "attribute" as key for nodes other than Root
    if discreteValue is not None:
        decisionTreeDict[node.getNode()][discreteValue] = newDecisionNode.getNode()

    # Split the vector feature data set
    newSubFeatureCollection = vectorSplit(columnarFeatureVector, columnarFeatureVector[-1], columnarFeatureVector[splitNodeIndex])
    
    sys.stdout.write("Number of Children:" + str(len(newSubFeatureCollection)) + "\n") if printDecisionTreeBuildProcessFlag == True else None
    sys.stdout.write("Splitting on :" + columnarFeatureVector[splitNodeIndex].getName() + "\n") if printDecisionTreeBuildProcessFlag == True else None
    
    # Print Node details
    for subFeatureBucket in newSubFeatureCollection:
        #print "DiscreteVal:", subFeatureBucket[0], "\t| ",
        sys.stdout.write("DiscreteVal:" + subFeatureBucket[0] + "\t| ") if printDecisionTreeBuildProcessFlag == True else None
        for feature in subFeatureBucket[1]:
            padding = " | "
            if feature.getCount() > 9:
                padding = "| "
            #print feature.getName(), "-", feature.getCount(), padding,
            sys.stdout.write(feature.getName() + "-" + str(feature.getCount()) + padding) if printDecisionTreeBuildProcessFlag == True else None
        sys.stdout.write("\n") if printDecisionTreeBuildProcessFlag == True else None

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
    global rootDecisionTree, printDecisionTreeBuildProcessFlag
    
    # Read the CSV data set file as vector
    vector = readFileAsVector(training_data)
    
    # Convert the data set into columnar format
    columnarVectorDataset = convertVectorToColumnar(vector)
    
    sys.stdout.write("\n"+"*"*80 + "\n\t\t\t  Building Decision Tree\n"+ "*"*80 +"\n") if printDecisionTreeBuildProcessFlag == True else None
    
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
    
    print "\n\n\n","*"*90
    print "\t\t\t\t Decision Tree"
    print "*"*90
    printTree(rootDecisionTree)
                 
def compute(training_data, test_data, depthLimitx, printBuildProcessFlag):
    ''' Main '''
    # Global declaration
    global rootDecisionTree,decisionTreeDict, depthLimit, printDecisionTreeBuildProcessFlag
    depthLimit = depthLimitx
    rootDecisionTree = None
    decisionTreeDict = {}
    printDecisionTreeBuildProcessFlag = printBuildProcessFlag
      
    # Train model
    trainModel(training_data)
    
    # Test model
    actualClassLabelList, predictedClassLabelList = testModel(test_data)
    
    return actualClassLabelList, predictedClassLabelList


            
if __name__ == '__main__':
    
    # Global variables
    rootDecisionTree = None
    decisionTreeDict = {}   # Master dictionary to maintain lookup path for nodes of decision tree
    depthLimit = 5          # Decision Tree depth
    printDecisionTreeBuildProcessFlag = True    # Flag to print decision tree build process

    training_data = "zoo-train.csv"
    test_data = "zoo-test.csv"
       
    compute(training_data, test_data, depthLimit, printDecisionTreeBuildProcessFlag )
    
    #printing decision tree built
    printDecisionTree()

