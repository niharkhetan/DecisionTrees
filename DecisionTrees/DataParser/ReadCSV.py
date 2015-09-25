'''
Created on Sep 18 , 2015

@author   : NiharKhetan, Ghanshyam Malu
@desc     : Reads a CSV dataset into vector format.

'''

import csv, sys
import os, Dataset


def readFileAsVector(csvFileName):
    ''' 
    Reads a data in CSV file and returns a vector dataset
    '''
    vector = []
    csvFileName = os.path.join(Dataset.__path__[0], csvFileName)
    try:
        with open(csvFileName, 'rb') as csvFile:
            reader = csv.reader(csvFile)
            try:
                for row in reader:
                    vector.append(row)
            except csv.Error as e:
                sys.exit('file %s, line %d: %s' % (csvFileName, reader.line_num, e))
    except IOError as e:
        sys.exit(e)
            
    return vector


def main():
    # Sample dataset to test the csv reader
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
        
    
if __name__ == '__main__':
    main()