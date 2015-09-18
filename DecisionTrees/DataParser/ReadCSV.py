'''
Created on Sep 18 , 2015

@author: Ghanshyam Malu
'''

import csv, sys

def readFileAsVector(csvFileName):
    ''' 
    Reads a data in CSV file and returns a vector dataset
    '''
    vector = []
    
    with open(csvFileName, 'rb') as csvFile:
        reader = csv.reader(csvFile)
        try:
            for row in reader:
                vector.append(row)
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (csvFile, reader.line_num, e))
        except IOError as e:
            sys.exit('Could not read file %s, line %d: %s' % (csvFile, reader.line_num, e))
            
    return vector

if __name__ == '__main__':
    training_data = "zoo-train.csv"
    vector = readFileAsVector(training_data)
    #print vector  
    print "Features:" 
    print '-'*60
    print "\t".join(vector[0]) 
    
    print "\n"
    print "Data:"
    print '-'*60 
    for row in vector[1:]:
        print  "\t".join(row) 
        
