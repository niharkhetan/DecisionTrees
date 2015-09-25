'''
Created on Sep 18, 2015

@author   : NiharKhetan, Ghanshyam Malu
@desc     : Feature class to hold the feature vector

'''
from collections import Counter

class Feature(object):
    '''
    This class represent each column in the dataset
    '''

    def __init__(self, name, data, isClassLabel = False):
        '''
        Constructor
        '''
        self.__name__ = name
        self.__data__ = data
        self.__count__ = len(data)
        self.__isClassLabel__ = isClassLabel
        self.__discreteSet__ = self.__calculateDiscreteSet__(data)
    
    def getName(self):
        return self.__name__
    
    def getData(self):
        return self.__data__
    
    def setData(self, data):
        self.__data__ = data
        self.__count__ = len(data)
        self.__discreteSet__ = self.__calculateDiscreteSet__(data)
        
    def getCount(self):
        return self.__count__
    
    def isClassLabel(self):
        return self.__isClassLabel__
    
    def getDiscreteSet(self):
        return self.__discreteSet__
    
    def __calculateDiscreteSet__(self, data):
        set = Counter(data)
        return set
    
if __name__ == '__main__':
    # Sample data to test the class
    f1 = Feature("Feature1", [1,1,1,1,1,1,1,1,0,0,3,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0])
    print f1.getName()   
    print f1.getCount()
    print f1.isClassLabel()
    print f1.getDiscreteSet()
    print f1.getDiscreteSet().keys()
    print f1.getDiscreteSet().values()
    print f1.getDiscreteSet()[f1.getDiscreteSet().keys()[0]]
    print f1.getDiscreteSet().keys()[0]
    print f1.getDiscreteSet().most_common(1)[0][0]