'''
Created on Sep 16, 2015

@author: NiharKhetan
'''

def readFileAsVector(fileName):
    vector = []
    fp = open(fileName)
    for eachLine in fp:        
        if (',' in eachLine):
            vector.append(commaSeperated(eachLine))
    return vector
    
def commaSeperated(eachLine):
    columns = eachLine.split(',')
    
    eachVector = doCleanup(columns)
    return eachVector

def doCleanup(columns):
    eachVector = []
    for eachColumn in columns:
        eachColumn = eachColumn.strip().strip('\n')        
        if eachColumn.isdigit():
            eachVector.append(int(eachColumn))
        else:
            eachVector.append(eachColumn)
    return eachVector

if __name__ == '__main__':
    vector = readFileAsVector("zoo-train.csv")
    print vector