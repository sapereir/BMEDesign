from __future__ import division
import sys
from math import log
import csv
import re

# if __name__ == '__main__':
#     input = sys.argv[1]
#     print("The input file is: %s" % (input)) 

class injectionInfo:
    def __init__(self, entry):
        self.injectorSN = entry[0]
        self.contrast = entry[1]
        self.catheter = entry[2]
        self.placement = entry[3]
        self.protocol = entry[4]
        self.injectionDate = entry[6]
        self.termination = self.setTermination(entry[7])
        self.transients = entry[8]
        self.pressureLimit = entry[9]
        self.time, self.pressure, self.rate = entry[10], entry[11], entry[12]
    
    def setTermination(self, term):
        if(term=="Normal"):
            return False
        else:
            return True
    

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
    return lines

lines = readFile("SN-SRMC-CT-1.txt")

#returns an array of injectionInfo objects created from each entry in the text file
def createObjects(lines):
    ret = []
    for l in lines:
        obj = injectionInfo(l)
        ret.append(obj)
    return ret

objs = createObjects(lines)
print(objs[4].rate)
        
        

