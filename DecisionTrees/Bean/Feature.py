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
    
    def getName(self):
        return self.__name__
    
    def getData(self):
        return self.__data__
    
    def getCount(self):
        return self.__count__
    
    def isClassLabel(self):
        return self.__isClassLabel__    