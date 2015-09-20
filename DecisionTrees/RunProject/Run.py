'''
Created on Sep 20, 2015

@author: NiharKhetan
'''
from TreeBuilder.Utils import *
from Evaluation.Metrics import *
def main():
    ''' Main '''
   
    #############################################################################
    #change depthLimit for different depth Limits
    #change training_data and test_data to put file names dircetlt
    #make sure data exists in Dataset folder
    #############################################################################  
    depthLim = 2
    training_data = "zoo-train_withoutFirstFeature.csv"
    test_data = "zoo-test_withoutFirstFeature.csv"
    
    actualClassLabelList, predictedClassLabelList = compute(training_data, test_data, depthLim)
    
    print "\n"*3
    
    #printing decision tree built
    printDecisionTree()
    
    #Error Rate
    findErrorRate(predictedClassLabelList, actualClassLabelList)
    
    #Accuracy
    findAccuracy(predictedClassLabelList, actualClassLabelList)
    
    #Confusion Matrix
    confusionMatrix = constructConfusionMatrix(predictedClassLabelList, actualClassLabelList)
    printConfusionMatrix(confusionMatrix)
        
if __name__ == '__main__':
    
    main()