#!/usr/bin/env python
# coding: utf-8

# # Initial Imports

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import time
import os


# In[2]:


cuda = torch.cuda.is_available()
print(cuda)


# # Loading Data

# In[3]:


train = np.load('train.npy', allow_pickle=True)
train_labels = np.load('train_labels.npy', allow_pickle=True)

dev = np.load('dev.npy', allow_pickle=True)
dev_labels = np.load('dev_labels.npy', allow_pickle=True)

test = np.load('test.npy', allow_pickle=True)


# # Want to work on Subset of Data??? 

# In[4]:


#size = 4096

# train = train[0:size]
# train_labels = train_labels[0:size]

# dev = train[0:size]
# dev_labels = train_labels[0:size]


# In[5]:


from torch.utils.data import DataLoader, Dataset, TensorDataset


# # Padding and Context

# In[6]:


class MyDataset(Dataset):
    def __init__(self, X, Y, context):
        self.context = context
        self.indexIJ = []
        for i in range(len(X)):
            currY = X[i]
            for j in range(len(currY)):
                self.indexIJ.append((i, j))
        for i in range(len(X)):
            lengthYDimension = np.shape(X[i])[0]
            npadX = ((context, context), (0, 0))
            X[i] = np.pad(X[i], npadX, mode='constant', constant_values=0)
            npadY = ((context, context))
            Y[i] = np.pad(Y[i], npadY , mode='constant', constant_values=0)
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.indexIJ)

    def __getitem__(self, index):
        indexI, indexJ = self.indexIJ[index]
        X = self.X[indexI][indexJ: indexJ + 2*self.context + 1] #flatten the input
        Y = self.Y[indexI][indexJ + self.context]
        return X.astype("float").reshape(-1), Y.astype("long")


# # Create Datasets

# In[7]:


num_workers = 4 if cuda else 0 

context = 12

# Training
train_dataset = MyDataset(train, train_labels, context)

train_loader_args = dict(shuffle=True, batch_size=1024, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=True, batch_size=64)
train_loader = DataLoader(train_dataset, **train_loader_args)

# Validation
dev_dataset = MyDataset(dev, dev_labels, context)

dev_loader_args = dict(shuffle=False, batch_size=1024, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=False, batch_size=1)
dev_loader = DataLoader(dev_dataset, **dev_loader_args)


# # Padding and Context for Testing

# In[8]:


class MyTestset(Dataset):
    def __init__(self, X, context):
        self.context = context
        self.indexIJ = []
        for i in range(len(X)):
            currY = X[i]
            for j in range(len(currY)):
                self.indexIJ.append((i, j))
        for i in range(len(X)):
            lengthYDimension = np.shape(X[i])[0]
            npadX = ((context, context), (0, 0))
            X[i] = np.pad(X[i], npadX, mode='constant', constant_values=0)
        self.X = X

    def __len__(self):
        return len(self.indexIJ)

    def __getitem__(self, index):
        indexI, indexJ = self.indexIJ[index]
        X = self.X[indexI][indexJ: indexJ + 2*self.context + 1] #flatten the input
        return X.astype("float").reshape(-1)


# # Create Model

# In[9]:


import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim


# In[10]:


class MyNetwork(nn.Module):
    def __init__(self, context_size, feature_size, hidden_size, output_size):
        super().__init__()
        input_size = (2*context_size + 1)*feature_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.b1 = nn.BatchNorm1d(hidden_size)
        self.dp1 = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.act2 = nn.ReLU()
        self.b2 = nn.BatchNorm1d(hidden_size//2)
        self.dp2 = nn.Dropout(p=0.2)
        self.layer3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.ac3 = nn.ReLU()
        self.b3 = nn.BatchNorm1d(hidden_size//4)
        self.dp3 = nn.Dropout(p=0.2)
        self.layer4 = nn.Linear(hidden_size//4, hidden_size//8)
        self.ac4 = nn.ReLU()
        self.b4 = nn.BatchNorm1d(hidden_size//8)
        self.dp4 = nn.Dropout(p=0.2)
        self.layer5 = nn.Linear(hidden_size//8, output_size)

    def forward(self, input_val):
        h = input_val
        h = self.layer1(h)
        h = self.act1(h)
        h = self.b1(h)
        h = self.dp1(h)
        h = self.layer2(h)
        h = self.act2(h)
        h = self.b2(h)
        h = self.dp2(h)
        h = self.layer3(h)
        h = self.ac3(h)
        h = self.b3(h)
        h = self.dp3(h)
        h = self.layer4(h)
        h = self.ac4(h)
        h = self.b4(h)
        h = self.dp4(h)
        h = self.layer5(h)
        return h


# # Define Model

# In[11]:


context = 12
features = 40
hidden_size = 4096
output_size = 138


# In[12]:


model = MyNetwork(context, features, hidden_size, output_size)
model = model.float()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
print(model)


# # Create Train Function

# In[13]:


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
        
    start_time = time.time()
    og_time = start_time
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device

        outputs = model(data.float())
        
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()
        
        loss = criterion(outputs, target)
        running_loss += loss.item()
        if (batch_idx%4096 == 0):
            end_time = time.time()
            print("Training:", "Batch Number " + str(batch_idx), "Running Loss:", running_loss, "Time:", end_time - start_time, 's')
            start_time = end_time
            
        loss.backward()
        optimizer.step()
    end_time = time.time()
    
    running_loss /= len(train_loader)
    acc = (correct_predictions/total_predictions)*100.0
    print('Training Accuracy: ', acc, 'Training Loss: ', running_loss, 'Time: ',end_time - og_time, 's')
    return running_loss


# # Create Validation Function

# In[14]:


def val_model(model, dev_loader, criterion):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(dev_loader):   
            data = data.to(device)
            target = target.to(device)

            outputs = model(data.float())

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
            
        running_loss /= len(dev_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Validation Accuracy: ', acc, '%', 'Validation Loss: ', running_loss)
        return running_loss, acc


# # Load Model?

# In[ ]:





# # Begin Training

# In[27]:


n_epochs = 65
Train_loss = []
Dev_loss = []
Dev_acc = []

for i in range(50, n_epochs):
    print("Epoch " + str(i)) 
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    dev_loss, dev_acc = val_model(model, dev_loader, criterion)
    Train_loss.append(train_loss)
    Dev_loss.append(dev_loss)
    Dev_acc.append(dev_acc)
    print('='*30)
    print("Saving the Model Version " + str(i))
    torch.save(model, './model_final' + str(i) +  '.pt')
    torch.save(optimizer, './adam_model_final' + str(i) +  '.pt')
    


# # Load Model?

# In[29]:


load = True
if load:
    model = torch.load("./model_final58.pt")
    optimizer = torch.load("adam_model_final58.pt")
    model.eval()


# # Testing Dataset

# In[30]:


# Testing
test = np.load('test.npy', allow_pickle=True)

test_dataset = MyTestset(test, context)

test_loader_args = dict(shuffle=False, batch_size=1, num_workers=num_workers, pin_memory=True) if cuda                    else dict(shuffle=False, batch_size=1)

test_loader = DataLoader(test_dataset, **test_loader_args)


# In[ ]:


import pandas as pd
model.eval()

predictions = np.zeros(len(test_loader))
for i, x in enumerate(test_loader):
    y = model(x.float().cuda())
    _, sm_test_output = torch.max(y, dim=1)
    predictions[i] = sm_test_output
df = pd.DataFrame()
df['id']= np.arange(len(predictions))
df['label'] = predictions.astype(int)
df.to_csv('For_kobe_24.csv', index=None)

assert(len(test_loader) == 223592)

