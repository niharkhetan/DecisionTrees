'''
Created on Sep 18, 2015

@author   : NiharKhetan, Ghanshyam Malu
@desc     : Node class to hold the nodes of the Decision Tree

        self.thisNode     - Node name
        self.parentNode   - parentNode
        self.children     - List of children nodes
        self.level        - level/depth of the node
        self.feature      - feature vector of the node
        self.attribute    - attribute of the feature
        self.leafFlag     - Flag to check if the node is leaf/label

'''
from Bean.Feature import Feature

class Node(object):
    '''
    classdocs
    '''
    
    def __init__(self, thisNode, attribute, feature, parentNode, children, level=0, leafFlag=False):
        '''
        Constructor
        '''
        self.thisNode = thisNode
        self.parentNode = parentNode
        self.children = children
        self.level = level
        self.feature = feature
        self.attribute = attribute
        self.leafFlag = leafFlag
        
  
    def getNode(self):
        return self.thisNode
    
    def getParentNode(self):
        return self.parentNode
        
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
        
    def removeChildNode(self,childNode):
        self.children.remove(childNode)
        
    def removeChildren(self):
        self.children = []