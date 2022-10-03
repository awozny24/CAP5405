# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:40:52 2022

@author: LuTimothy
"""
import os
import re
import numpy as np
#import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
#resource: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/

#Neural Net parameter
input_dim = 42
#technically hidden_dim will be same size as batch_size
hidden_dim = 200
output_dim = 1


batch_size = 200
iterations = 400
num_epochs = 5

def convert():
    path = os.getcwd()
    file = path + '//connectfour.data'
    #regexp 
    outcomes = 'win|loss|draw'
    
    snapShots = np.zeros((6,7,1))
    gameStates =np.zeros((1))
    firstLine = True
    index = 1
    NumSamples = 40000
    
    with open(file, mode ='r') as f:
        for line in f:
            line = line.strip()
            if index == NumSamples + 1:
                break
            
            match = re.search(outcomes,line)
            try:
               board, state = re.split(outcomes,line)
            except ValueError:
                   print(f"error occured on line {index}")
        
            #convert comma separated string to list
            board = board.split(',')
     
            #remove empty values from board
            board = [x for x in board if x]    
            board = np.array(list(board)).reshape(7,6)
        
            #reorient
            board = np.flipud(board.T)
            #add axis
            board = board[:,:,np.newaxis]
            state = match.group()
            state = np.array(state)
       
            if firstLine:
                snapShots = board
                gameStates = state
            else:
                snapShots = np.concatenate((snapShots, board), axis = 2)
                gameStates = np.hstack((gameStates, state))
        
            firstLine = False
           # print(f"this is line number {index} \n")
            index += 1
        # = np.array(list(board)).reshape(6,7)
        with open('snapShots.npy', 'wb') as f:
            np.save(f, snapShots)
        with open('gameStates.npy', 'wb') as f:
            np.save(f, gameStates)
        
    return snapShots, gameStates

def load():
    snapShots = np.load('snapShots.npy')
    gameStates= np.load('gameStates.npy')
    
    #replace values in snapShots and gameState to int
    snapShots = np.where(snapShots == 'b', 0, snapShots)
    snapShots = np.where(snapShots ==  'x', 1, snapShots)
    snapShots = np.where(snapShots == 'o', -1, snapShots)
    snapShots = snapShots.astype(int)
    
    gameStates = np.where(gameStates =='win', 1, gameStates)
    gameStates = np.where(gameStates =='draw', 0, gameStates)
    gameStates = np.where(gameStates =='loss', -1, gameStates)
    gameStates = gameStates.astype(int)
    
    
    return snapShots, gameStates

#saves the trained feed-forward neural network
def save(model):
    PATH = 'FFNN_MODEL.pt'
    torch.save(model, PATH)
    
    

class customDataset(Dataset):
    def __init__(self, X, y):
        #normalize inputs
        #!!! important
        X = torch.from_numpy(X).float()
        X = normalize(X)
        y = torch.from_numpy(y)
        self.indices = len(y)
        
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.indices
    
    def __getitem__(self, idx):
       # board_tensor = torch.from_numpy(self.X)
        #state_tensor = torch.from_numpy(self.y)
       # print(idx)
        return self.X[:,:,idx], self.y[idx]
        
    

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 

        # Non-linearity
        #don't use sigmoid
        self.relu = nn.ReLU()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.relu(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out
    


#only need to call convert once
#convert()



snapShots, gameStates = load()

#split into training and test
all_indices = list(range(20000))
train_ind, test_ind = train_test_split(all_indices, test_size=0.3)

#can't directly call train test split on data as it causes problems
X_train = snapShots[:,:,train_ind]
y_train = gameStates[train_ind]

X_test = snapShots[:,:,test_ind]
y_test = gameStates[test_ind]

#create datasets for train and test
train_dataset = customDataset(X_train, y_train)
len(train_dataset)
test_dataset = customDataset(X_test, y_test)

#create datasets
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
#using negative log likelihood instead of cross entropy
criterion = nn.MSELoss()


count = 0
#train the neural network here
for epoch in range(num_epochs):
    for i, (boards, states) in enumerate(train_loader):
        boards = torch.tensor(boards, dtype = float, requires_grad = True)
       
     

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        #you probably need to reshape boards
        #batch size, row length, column length
        #something like batch,size, dimensions
        
        boards= boards.reshape(200,-1).float()
        outputs = model(boards)

        #negative log likelihood
        #drop redundant dimension in outputs
        loss = criterion(outputs.squeeze(), states.float())

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()
       # print(f"index i is {i}")
        count += 1

        if count % 200 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            accuracy = 0
            # Iterate through test dataset
            #not evaluating over every batch
            for boards, states in test_loader:
                boards = torch.tensor(boards, dtype = float, requires_grad = True)
              
              
                boards= boards.reshape(200,-1).float()
                # Forward pass only to get logits/output
                outputs = model(boards)
                #outputs are probabilities
                outputs = torch.round(outputs)
                

                # # Get predictions from the maximum value
                # _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += states.size(0)

                # Total correct predictions
                #convert states to correct dtype
                states = states.type(torch.int64)
                correct += (outputs.squeeze() == states).sum()

            accuracy =  correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(count, loss.item(), accuracy))
            
#save the trained model
save(model)