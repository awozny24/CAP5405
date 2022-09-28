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


#Neural Net parameter
input_size = 42
hidden_size = 800
output_size = 42



def convert():
    path = os.getcwd()
    file = path + '//connectfour.data'
    #regexp 
    outcomes = 'win|loss|draw'
    
    snapShots = np.zeros((6,7,1))
    gameStates =np.zeros((1))
    firstLine = True
    index = 1
    NumSamples = 2000
    
    with open(file, mode ='r') as f:
        for line in f:
            line = line.strip()
            if index == NumSamples:
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
    gameState = np.load('gameStates.npy')
    
    return snapShots, gameState

#only need to call convert once
convert()