'''
Created on Sep 18, 2015

@author: NiharKhetan
'''

class Node(object):
    '''
    classdocs
    '''
    

    def __init__(self, thisNode, children, level=0):
        '''
        Constructor
        '''
        self.thisNode = thisNode
        self.children = children
        self.level = level
        