
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import time
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
import sys
from math import log
import csv
import re


#initializing values used to enumerate string variables in input csv files
#there may be a better way to enumerate but I just went with this method I wrote
contrasts=dict()
contrastCount = 0
catheters = dict()
catheterCount = 0
placements = dict()
placementCount = 0
protocols = dict()
protocolCount = 0
transients = dict()
transientCount = 0

class MyDataset(Dataset):
    def __init__(self, context):
        #self.allValues = tuple full of tuples representing each row in the csv file
        self.allValues = []
        #original and raw data parsed from csv file as an array
        self.context = context
        #variables to keep track of the current number of occurances of unique string variables for a given category
        #in the dictionary
        global contrastCount,catheterCount, placementCount, protocolCount, transientCount
        #loop used to iterate over each row and enumerate values that are strings
        for row in context:
            #enumerating all strings in contrast, catheter, placement, protocol, transient to numbers
            #converting setTermination from string to integer 
            row[1], contrastCount = self.convertIntoBinary(contrasts, contrastCount,row[1])
            row[2],catheterCount =self.convertIntoBinary(catheters, catheterCount,row[2])
            row[3], placementCount = self.convertIntoBinary(placements, placementCount, row[3])
            row[4], protocolCount = self.convertIntoBinary(protocols, protocolCount, row[4])
            row[8], transientCount = self.convertIntoBinary(transients, transientCount, row[8])
            row[7] = self.setTermination(entry[7])
            #convert the row from list to immutable tuple
            #append to all values
            self.allValues.append(tuple(row))
        #convert list of all rows into a tuple
        self.allValues = tuple(self.allValues)

        #primarily used in __getItem__ to index into data set
        self.indexIJ = []
        for row in range(len(self.allValues)):
            currY = X[i]
            for col in range(len(currY)):
                self.indexIJ.append((row, col))

        ## i'm not quite sure why np.pad is needed in your case, so i didn't address it
        # for i in range(len(X)):
        #     lengthYDimension = np.shape(X[i])[0]
        #     npadX = ((context, context), (0, 0))
        #     X[i] = np.pad(X[i], npadX, mode='constant', constant_values=0)
        #     npadY = ((context, context))
        #     Y[i] = np.pad(Y[i], npadY , mode='constant', constant_values=0)

    #used to convert termination from string to integer (or boolean if you change the #'s)
    #returns integer enumeration of value
    def setTermination(self, term):
        if(term=="Normal"):
            return 0
        else:
            return 1

    #used to enumerate the strings of certain columns into corresponding integers
    #returns the value (enumeration) of that string in the dict and the current # of occurances of that variable 
    def convertIntoBinary(dic, count, var):
        if var not in dic.keys():
            dic[var] = count+1
            count += 1
        return dic[var], count

    def __len__(self):
        return len(self.allValues)

    def __getitem__(self, index):
        indexI, indexJ = self.indexIJ[index]
        #i'm not sure why you flattened the input here... 
        #i implemented normal indexing into a tuple without accounting for flattening/padding
        # X = self.X[indexI][indexJ: indexJ + 2*self.context + 1] #flatten the input
        # Y = self.Y[indexI][indexJ + self.context]
        return self.allValues[indexI][indexJ]

#reads the CSV file
#cleans the file by removing tabs/punctuation
#returns an array of lists (2D array) representing each row in the file
def readFile(path):
    lines = list()
    with open(path) as file:
        csv_reader = file.readlines()
        line_count = 0
        for row in csv_reader:
            row = re.split(r'\t+', row.rstrip('\t'))[1:]
            values = row.pop(-1)
            values = values.split(';')
            for r in values:
                chunk = re.findall('\d*\.?\d+',r)
                chunk = list(map(float, chunk)) 
                row.append(chunk)
            lines.append(row)
    return lines, ret


#variables to run each function
original = readFile("../SN-SRMC-CT-1.txt")
dataset = MyDataset(original)
