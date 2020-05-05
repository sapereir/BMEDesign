#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import time
import os
import re


# In[2]:


cuda = torch.cuda.is_available()
print(cuda)


# In[3]:


from torch.utils.data import DataLoader, Dataset, TensorDataset

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.optim as optim


# In[ ]:





# In[4]:


class MyDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        X, Y = self.values[index]
        return np.array(X).astype("float"), np.array(Y).astype("long")


# In[5]:


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
    features = []
    for x in lines:
        features.append([x[2], x[3], x[4], max(x[10])])
    filterUnk = list(filter(lambda x: 'Unknown' not in x, features))
    notGauges = ['PowerPICC', 'PICC', 'PowerPort', 'CentralLine', 'Other']
    notivLoc = ['Other', 'PowerPort', 'CentralLine', 'PICC', 'PowerPICC']
    filterGauge = list(filter(lambda x: x[0] not in notGauges, filterUnk))
    filterIV = list(filter(lambda x: x[1] not in notivLoc, filterGauge))
    return filterIV


# In[6]:

original = readFile("SN-SRMC-CT-1.txt") 

def parseData():
    # original = readFile("SN-SRMC-CT-1.txt") + readFile("SN-SRMC-CT-2.txt")
    gaugeTypes = set()
    ivLocation = set()
    protocol = dict()
    pressure = set()
    for inputX in original:
        gaugeTypes.add(inputX[0])
        ivLocation.add(inputX[1])
        if inputX[2] in protocol:
            protocol[inputX[2]] = protocol[inputX[2]] + [inputX[3]]
        else:
            protocol[inputX[2]] = [inputX[3]]
        pressure.add(inputX[3])
    return gaugeTypes, ivLocation, protocol, pressure


# In[8]:


def parseProtocol(x):
    protocol = x.split(" ")
    Contrast = 0
    Saline = 0
    Mixed = 0
    AmountC = 0
    AmountS = 0
    AmountM = 0
    PercentM = 0
    Flowrate = 0
    for i in protocol:
        if i[0] == "@":
            FlowRate = float(i[1:])
        if i[0] == "C":
            Contrast = 1
            AmountC = int(i[1:])
        if i[0] == "S":
            Saline = 1
            AmountS = int(i[1:])
        if i[0] == "R":
            rprot = i.split("%")
            Mixed = 1
            AmountM = int(rprot[0][1:])
            PercentM = int(rprot[1][1:])
    return [Contrast, Saline, Mixed, AmountC, AmountS, AmountM, PercentM, FlowRate]
    print(v)


# In[9]:


def roundup(x):
    return int(math.ceil(x/100.0))*100

buckets = []
def organizeData():
    gaugeTypes, ivLocation, protocol, pressure = parseData()
    mapGauge = dict()
    for i, x in enumerate(sorted(list(gaugeTypes))):
        mapGauge[x] = i
    mapIV = dict()
    for i, x in enumerate(sorted(list(ivLocation))):
        mapIV[x] = i
    mapPressure = dict()
    count = 0
    bucketSize = 20
    for p in range(int(min(pressure)), int(roundup(max(pressure))), bucketSize):
        buckets.append(p)
        mapPressure[p] = count
        count += 1
    result = []
    for v in original: 
        for p in range(len(buckets) - 1):
            if (v[3] >= buckets[p]) and (v[3] < buckets[p + 1]):
                pressureV = buckets[p]
        result.append(([mapGauge[v[0]], mapIV[v[1]]] + parseProtocol(v[2]) , mapPressure[pressureV]))
    return result

# In[10]:

result = organizeData()
half = len(result)//2

training_set = result[:half]
validation_set = result[half:]


# In[11]:


num_workers = 8 if cuda else 0 

train_dataset = MyDataset(training_set)
train_loader_args = dict(shuffle=True, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=True, batch_size=64)
train_loader = DataLoader(train_dataset, **train_loader_args)

val_dataset = MyDataset(validation_set)
val_loader_args = dict(shuffle=False, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=False, batch_size=1)
val_loader = DataLoader(val_dataset, **val_loader_args)


# In[12]:


input_size = len(train_dataset.__getitem__(0)[0])
output_size = len(buckets)
hidden_size = int(input_size**2)//2

input_size, output_size, hidden_size


# In[13]:


class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.b1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, int(hidden_size))
        self.act2 = nn.ReLU()
        self.b2 = nn.BatchNorm1d(int(hidden_size))
        self.layer3 = nn.Linear(int(hidden_size), output_size)

    def forward(self, input_val):
        h = input_val
        h = self.layer1(h)
        h = self.act1(h)
        h = self.b1(h)
        h = self.layer2(h)
        h = self.act2(h)
        h = self.b2(h)
        h = self.layer3(h)
        return h


# In[14]:


model = MyNetwork(input_size, hidden_size, output_size)
model = model.float()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
device = torch.device("cpu" if cuda else "cpu")
model = torch.load("c_model.pt")
model.to(device)
print(model)
model.eval()

class MyDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        X = self.values[index]
        return np.array(X).astype("float")
        
constant = 72
# function that is called by gui to get the maxPressure value
# not sure where to put it so i decided to put it after data parsing
def predictMaxPressure(protocol):
    d = MyDataset([protocol])
    d_args = dict(shuffle=True, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    d_loader = DataLoader(d, **d_args)
    for y in d_loader:
      y.to(device)
      outputs = model(y.float())
      _, predicted = torch.max(outputs.data, 1)
    return predicted.item()*20*constant

protocol, pressure = result[0]
pressure = pressure*constant

p = predictMaxPressure(protocol)

print("Pressure:", p)