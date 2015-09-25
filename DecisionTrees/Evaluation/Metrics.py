'''
Created on Sep 20, 2015

@author   : NiharKhetan, Ghanshyam Malu
@desc     : Methods to compute accuracy, error rate, construct Confusion Matrix.

'''

#from sklearn.metrics import confusion_matrix
from collections import Counter

def findAccuracy(predicted, expected):
    '''
    Returns the Accuracy which is correctly predicted by total 
    @param predicted: type(list) list of predicted values
    @param expected: type(list) list of expected values
    '''
    print "\n","="*90
    print "\t\t\t\t\tACCURACY"
    print "="*90
    correct = 0
    for i in range(0, len(predicted)):
        if predicted[i] == expected[i]:
            correct += 1
    accuracy = correct / float(len(predicted)) * 100
    print "\n\tIncorrect Classification Count: %d \tCorrect Classification Count: %d" %(len(predicted) - correct, correct)
    print "\n\t","*" * 73
    print "\t\t\t\tAccuracy is %.3f %%" % (accuracy)
    print "\t","*" * 73
    print "\n"*2

def findErrorRate(predicted, expected):
    '''
    Returns the ErrorRate which is incorrectly predicted by total 
    @param predicted: type(list) list of predicted values
    @param expected: type(list) list of expected values
    '''
    print "\n","="*90
    print "\t\t\t\t\tERROR RATE"    
    print "="*90
    incorrect = 0
    for i in range(0, len(predicted)):
        if predicted[i] != expected[i]:
            incorrect += 1
    errorRate = incorrect / float(len(predicted)) * 100
    print "\n\tIncorrect Classification Count: %d \tCorrect Classification Count: %d" %(incorrect, len(predicted) - incorrect)
    print "\n\t","*" * 73
    print "\t\t\t\tError Rate is %.3f %%" % (errorRate)
    print "\t","*" * 73
    print "\n"*2


def indexListForItem(item, inputList):    
    list  = [i for i,x in enumerate(inputList) if x == item]
    return list

def findMatches(item, indexList, predicted):
    '''
    find count of an item in another list
    @param item: item which needs to be looked for in the predicted list
    @param indexList: list of indexes where given items need to be ooked upon in the predicted list
    @param predicted: list of predicted items
    @return count of items matched at the index    
    '''
    count = 0    
    for eachIndex in indexList:
        if item == predicted[eachIndex]:
            count += 1
    return count        

def constructConfusionMatrix(predicted, expected):
    '''
    Constructs the confusion matrix
    @param predicted: list of predicted items
    @param expected: list of expected items
    @return cm: type(dict) the confusion matrix   
    '''
    expectedCounter = Counter(expected)
    cmatrix = {}
        
    #creating an empty matrix
    for eachKey in expectedCounter.keys():
        cmatrix[eachKey] = []
        for i in range(len(expectedCounter.keys())):
            cmatrix[eachKey].append(None)
    
    #filling out values in the matrix
    for i in range(0,len(expectedCounter.keys())):
        indexListOfKey = indexListForItem(expectedCounter.keys()[i], expected)
        for j in range(len(expectedCounter.keys())):                       
            ct = findMatches(expectedCounter.keys()[j], indexListOfKey, predicted)            
            cmatrix[expectedCounter.keys()[i]][j] = ct          
    return cmatrix

def printConfusionMatrix(cmatrix):
    '''
    prints the confusion matrix
    @param cmatrix: type(dict)
    '''
    
    print "\n","="*90
    print "\t\t\t\tCONFUSION MATRIX"
    print "="*90
    print "\n"
    print '*'*30, 'Predicted Values As Columns', '*'*30
    print '*'*30, 'Expected Values As Rows    ', '*'*30
    print '\n\t\t:',
    for key in cmatrix.keys():
        print '\t',key,'\t|',
    print ''
    print '-'*16,'-'*16*len(cmatrix.keys())
    for key in cmatrix.keys():
        print '\t',key,'\t:',
        for value in cmatrix[key]:
            print '\t',value,'\t|',
        print ''            


if __name__ == '__main__':
    # Sample data to test the metrics
    predicted = [1,1,4]
    expected = [1,0,4]
    predicted = [0, 0, 2, 2, 0, 2, 1]
    expected = [2, 0, 2, 2, 0, 1, 0]
    
    predicted = ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '2', '2', '2', '3', '3', '4', '4', '4', '4', '5', '6', '6', '6', '7', '7', '7', '7']
    expected = ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '2', '2', '2', '7', '5', '4', '4', '4', '4', '5', '6', '6', '6', '7', '7', '1', '7']
    
    for i in range(0,len(predicted)):
        predicted[i] = int(predicted[i])
        expected[i] = int(expected[i])

    findAccuracy(predicted, expected)
    findErrorRate(predicted, expected)
    #cm = confusion_matrix(expected , predicted)

    confusionMatrix = constructConfusionMatrix(predicted, expected)

    printConfusionMatrix(confusionMatrix)
    
    '''
    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')    
    plt.show()
    '''