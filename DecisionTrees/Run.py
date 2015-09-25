'''
Created on Sep 20, 2015

@author   : NiharKhetan, Ghanshyam Malu
@desc     : Core file to run the Decision Tree Modeler
            Train the model, Test it and plot the confusion matrix
            Also, visualizes it using the matplotlib.pyplot 
@Usage    : Execute the python file to run the Decision Tree Modeler
            $ python Run.py

@Version  : Uses Python 2.7

==========================================================================================
            Welcome to Decision Tree Modeler
==========================================================================================

Choose one of the available options:

    0 - Model, Train and Test 'Zoo' data
    1 - Model, Train and Test 'Carvana' data (personal dataset) 

'''
from TreeBuilder.BuildDecisionTree import *
from Evaluation.Metrics import *
#import matplotlib.pyplot as plt #Uncomment this line to visualize the confusion matrix using matplotlib

def getUserInput(msg, inputType, options = []):
    ''' Generalized method to get user input '''

    while True:
        try:
            userOption = raw_input(msg).upper()
            if inputType == "int":
                userOption = int(userOption)
            if len(options) > 0 and userOption not in options:
                raise ValueError('Invalid choice !') 
        except: 
            print "\nInvalid choice !"
        else:
            break
    
    return userOption

def getUserDatasetChoice(datasetListDict):
    ''' Display available datasets to user'''
            
    print "=" * 90
    print "\t\t\tWelcome to Decision Tree Modeler"
    print "=" * 90

    msg = "\nChoose one of the available options:\n\n"
    msg += "\t0 - Model, Train and Test 'Zoo' data\n"
    msg += "\t1 - Model, Train and Test 'Carvana' data (personal dataset) \n\n" 
    msg += "Your choice...(0 or 1):  "
    
    userOption = getUserInput(msg, "int", datasetListDict.keys())
    return userOption

def main():
    ''' Main '''
   
    #############################################################################
    # Change depthLimit for different depth Limits
    # Change training_data and test_data to put file names directly
    # Make sure data exists in Dataset folder
    #############################################################################  
    
    depthLim =3   # Decision Tree depth
        
    printDecisionTreeBuildProcessFlag = False    # Flag to print decision tree build process
    decisionTreeFinalDepth = 0                  # Final depth of the built decision tree
    
    datasetListDict = {0: ["zoo-train.csv","zoo-test.csv"],
                   1: ["carvana_train.csv", "carvana_test.csv"]}

    userDatasetChoice = getUserDatasetChoice(datasetListDict)

    training_data = datasetListDict[userDatasetChoice][0]
    test_data =  datasetListDict[userDatasetChoice][1]
    
    userInput = getUserInput("\nChange default settings for Max Tree Depth, Display log (Y/N)?... ", "char", ['Y','N'])

    if userInput == 'Y':
        
        userInput = getUserInput("\nDo you wish to change the default Max Depth ("+ str(depthLim)+ ") allowed for the Decision Tree? (Y/N)... ", "char", ['Y','N'])
        if userInput == 'Y':
                userInput = getUserInput("\nEnter the Max Depth for Decision Tree... ", "int", [])
                if userInput < 1:
                    print "Invalid depth specified, using default Max Depth :", depthLim
                else:
                    depthLim = userInput
        
        userInput = getUserInput("\nDo you wish to print detailed log for Decision Tree modeling process? (Y/N)... ", "char", ['Y','N'])        
        if userInput == 'Y':
            printDecisionTreeBuildProcessFlag = True
        
    actualClassLabelList, predictedClassLabelList = compute(training_data, test_data, depthLim, printDecisionTreeBuildProcessFlag, decisionTreeFinalDepth)
    
    #printing decision tree built
    printDecisionTree()
    
    #Error Rate
    findErrorRate(predictedClassLabelList, actualClassLabelList)
    
    #Accuracy
    findAccuracy(predictedClassLabelList, actualClassLabelList)
    
    #Confusion Matrix
    confusionMatrix = constructConfusionMatrix(predictedClassLabelList, actualClassLabelList)
    printConfusionMatrix(confusionMatrix)

    
    print "\n\n","-" * 90
    print "\t\tThank you for using the Decision Tree Modeler !"
    print "-" * 90,"\n\n"
   
    
    #################################################################################
    # IF SCIKIT IS INSTALLED UNCOMMENT THIS TO GET A CONFUSOIN MATRIX PLOT          #
    #################################################################################
    # Show confusion matrix in a separate window
    '''
    cm = confusion_matrix(actualClassLabelList , predictedClassLabelList)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')    
    plt.show()
    '''
    
        
if __name__ == '__main__':
    
    main()