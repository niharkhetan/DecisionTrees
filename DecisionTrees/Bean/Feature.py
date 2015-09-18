'''
Created on Sep 18, 2015

@author: NiharKhetan
'''

class Feature(object):
    '''
    classdocs
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
    
    def getCount(self):
        return self.__count__
    
    def isClassLabel(self):
        return self.__isClassLabel__
    
    def getDiscreteSet(self):
        return self.__discreteSet__
    
    def __calculateDiscreteSet__(self, data):
        discreteSet = []
        for eachType in data:
            if eachType not in discreteSet:
                discreteSet.append(eachType)
        return discreteSet
    
if __name__ == '__main__':
    f1 = Feature("Feature1", [1,1,1,1,1,1,1,1,0,0,3,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0])
    print f1.getName()
    print f1.getData()
    print f1.getCount()
    print f1.isClassLabel()
    print f1.getDiscreteSet()