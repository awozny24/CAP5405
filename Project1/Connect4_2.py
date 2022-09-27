# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:36:28 2022

@author: LuTimothy
"""

import os
import re
import numpy as np

#custom exception
class game_over(Exception):
    pass

COLUMNS = 7
ROWS = 6
FIRST = '1'
SECOND = '2'


def GetData(path = None):
    # get Connect4 data as a list
    path = os.getcwd()
    data = []
    with open(path + '//connectfour.data', mode = 'r') as infile:
        data.append([line.strip().split(',') for line in infile.readlines()])

    # get data as numpy array
    data = np.asarray(data)
    data = data[0]

    data[np.where(data=='b')] = 0
    data[np.where(data=='x')] = 1
    data[np.where(data=='o')] = -1
    data[np.where(data=='draw')] = 0
    data[np.where(data=='win')] = 1
    data[np.where(data=='loss')] = -1

    data = data.astype(np.float)

    return data[:, :-1], data[:, -1]


def load():
    path = os.getcwd()
    file = path + '//connectfour.data'
    #regexp 
    outcomes = 'win|loss|draw'
    
    snapShots = np.zeros((6,7,1))
    gameStates =np.zeros((1))
    firstLine = True
    index = 1
    NumSamples = 5000
    
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
    return snapShots, gameStates
        


class Game():
    def __init__ (self):
         self.win = 4
         self.board = np.zeros((ROWS,COLUMNS))
         
    def reset(self):
        self.board = np.zeros((ROWS,COLUMNS))
    
    def print_board(self):
        for i in range(ROWS):
            print(self.board[i,:])
    
    def available_space(self):
        if np.where(self.board== '0'):
            return True
        else:
            return False
        
         
    def valid_choice(self, col):
        if(col>=0 and col<=6):
            #check top of column
            if(self.board[0][col] == 0):
                return True
        return False
        
         
    def insert(self, column, mark):
        c = self.board[:,column]
        if self.valid_choice(column):
            #search from bottom to top
            i = -1
            while c[i] != 0:
                i -=1
            c[i] = mark
            return True
        else:
            return False
        
    def winning_move(self, mark):
        #Horizontal
        for c in range(COLUMNS-3):
            for r in range(ROWS):
                #use set operation to find winning combination
                #uncommenting blank prints out the space being searched on the board
                blank = np.zeros((6,7))
                blank[r,c:c+4] = '1'
                #print(blank)
                if list((set(self.board[r,c:c+4]))) == [mark]:
                    return True
        #Vertical
        for c in range(COLUMNS):
            for r in range(ROWS -3):
                #use set operation to find winning match
                blank = np.zeros((6,7))
                blank[r:r+4,c] = '1'
               # print(blank)
                if list(set(self.board[r:r+4,c])) == [mark]:
                    return True
                
             
        #Downwards Diag
        for c in range(COLUMNS-3):
            for r in range(ROWS -3):
                blank = np.zeros((6,7))
                #combine coordinates
                #here r and c are the offsets
                i = [r + a for a in range(COLUMNS-3)]
                j = [c + b for b in range(COLUMNS-3)]
                
                coord = [i,j]
                blank[coord[0],coord[1]] ='1'
               # print(blank)
                if list(set(self.board[coord[0],coord[1]])) == mark:
                    return True
               
        #Upwards Diag
        for c in range(COLUMNS -3):
            for r in range(3, ROWS):
                blank = np.zeros((6,7)) 
                #combine coordinates
                #here r and c are the offsets
                i = [r - a for a in range(COLUMNS-3)]
                j = [c + b for b in range(COLUMNS-3)]
                coord = [i,j]
                blank[coord[0],coord[1]] ='1'
               
               # print(blank)
                if list(set(self.board[coord[0],coord[1]])) == mark:
                    return True
                
        return False
                                                              
if __name__ == '__main__':
    g = Game()
    mark = FIRST
    #GetData()
   ## load()
    print("done loading")
    
    while ( True):
        g.print_board()
        val = input(f"player {mark}'s turn, enter column: \n")
        g.insert(int(val),mark)
        #evaluate winning move 
        if (g.winning_move(mark)):
            print(f"player {mark} has won the game!")
            break
        if mark == FIRST:
            mark = SECOND
        else:
            mark = FIRST
        if(g.winning_move(mark) == False and g.available_space() == False):
            print("game ended in a draw")
            break
        
    
    
	
        #val = input(f"player {mark}'s turn")
# 		row = input('{}\'s turn: '.format('Red' if turn == RED else 'Yellow'))
# 		g.insert(int(row), turn)
# 		turn = YELLOW if turn == RED else RED                
               
 
    # Check vertical locations for win
    # for c in range(COLUMN_COUNT):
    #     for r in range(ROW_COUNT-3):
    #         if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
    #             return True
 
    # # Check positively sloped diaganols
    # for c in range(COLUMN_COUNT-3):
    #     for r in range(ROW_COUNT-3):
    #         if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
    #             return True
 
    # # Check negatively sloped diaganols
    # for c in range(COLUMN_COUNT-3):
    #     for r in range(3, ROW_COUNT):
    #         if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
    #             return True
 
            
        
       
            
