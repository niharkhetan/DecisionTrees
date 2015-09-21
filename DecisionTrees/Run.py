'''
Created on Sep 20, 2015

@author: NiharKhetan
'''
from TreeBuilder.BuildDecisionTree import *
from Evaluation.Metrics import *
#import matplotlib.pyplot as plt

def main():
    ''' Main '''
   
    #############################################################################
    #change depthLimit for different depth Limits
    #change training_data and test_data to put file names dircetlt
    #make sure data exists in Dataset folder
    #############################################################################  
    depthLim = 3
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
    
    cm = confusion_matrix(actualClassLabelList , predictedClassLabelList)
    
    #################################################################################
    # IF SCIKIT IS INSTALLED UNCOMMENT THIS TO GET A CONFUSOIN MATRIX PLOT          #
    #################################################################################
    # Show confusion matrix in a separate window
    '''
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')    
    plt.show()
    '''
    
        
if __name__ == '__main__':
    
    main()