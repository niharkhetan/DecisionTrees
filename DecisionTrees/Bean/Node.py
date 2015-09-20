'''
Created on Sep 18, 2015

@author: NiharKhetan
'''
from Bean.Feature import Feature

class Node(object):
    '''
    classdocs
    '''
    
    def __init__(self, thisNode, attribute, feature, children, level=0, leafFlag=False):
        '''
        Constructor
        '''
        self.thisNode = thisNode
        self.children = children
        self.level = level
        self.feature = feature
        self.attribute = attribute
        self.leafFlag = leafFlag
  
    def getNode(self):
        return self.thisNode
        
    def getFeature(self):
        return self.feature


    def getChildren(self):
        return self.children
    
    def getLevel(self):
        return self.level
    
    def getAttribute(self):
        return self.attribute
    
    def isLeaf (self):
        return self.leafFlag
    
    def addChildNode(self,childNode):
        self.children.append(childNode)