from __future__ import division
import sys
from math import log
import csv
import re
import torch

# if __name__ == '__main__':
#     input = sys.argv[1]
#     print("The input file is: %s" % (input)) 

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

class injectionInfo:
    def __init__(self, entry):
        global contrastCount,catheterCount, placementCount, protocolCount, transientCount
        self.injectorSN = entry[0]
        self.contrast,contrastCount = convertIntoBinary(contrasts, contrastCount,entry[1])
        self.catheter,catheterCount =convertIntoBinary(catheters, catheterCount,entry[2])
        self.placement, placementCount = convertIntoBinary(placements, placementCount, entry[3])
        self.protocol, protocolCount = convertIntoBinary(protocols, protocolCount, entry[4])
        self.injectionDate = entry[6]
        self.termination = self.setTermination(entry[7])
        self.transient, transientCount = convertIntoBinary(transients, transientCount, entry[8])
        self.pressureLimit = entry[9]
        self.time, self.pressure, self.rate = entry[10], entry[11], entry[12]
    
    def setTermination(self, term):
        if(term=="Normal"):
            return False
        else:
            return True
    
    def getMaxPressure(self):
        return max(self.pressure)

def readFile(path):
    lines = list()
    ret = list()
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
            obj = injectionInfo(row)
            ret.append(obj)
    return lines, ret


def convertIntoBinary(dic, count, var):
    if var not in dic.keys():
        dic[var] = count+1
        count += 1
    return dic[var], count

def printInjectionInfo(info):
    print(str(info.injectorSN) + " " + str(info.contrast) + " " + str(info.catheter) + " " + str(info.placement) + " " + str(info.protocol) + " " + str(info.injectionDate) + " " +str(info.termination) + " " + str(info.transient) + " "+ str(info.pressureLimit) + " " + str(info.time))


lines,parsed = readFile("SN-SRMC-CT-1.txt")

print(contrasts)
print(transients)
print(lines[1])
printInjectionInfo(parsed[1])


        
        

