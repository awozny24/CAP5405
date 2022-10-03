# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:36:28 2022

@author: LuTimothy
"""

import os
import re
import numpy as np
import random
import Connect4FFNN 
import torch
import copy

#custom exception
class game_over(Exception):
    pass

COLUMNS = 7
ROWS = 6
FIRST = 1
SECOND = 2

input_dim = 42
hidden_dim = 200
output_dim = 1
        


class Game():
    def __init__ (self):
         self.board = np.zeros((ROWS,COLUMNS))
         #initialize FFNN model here
         PATH = 'FFNN_MODEL.pt'
         model = torch.load(PATH)
         model.eval()
         self.model = model
         
    def reset(self):
        self.board = np.zeros((ROWS,COLUMNS))
    
    def print_board(self):
        for i in range(ROWS):
            print(self.board[i,:])
    #need to fix
    def available_space(self):
        blanks = np.where(self.board== 0)
        if len(blanks[0]) != 0:
            return True
        else:
            return False
        
         
    def valid_choice(self, col):
        if(col>=0 and col<=6):
            #check top of column
            if(self.board[0][col] == 0):
                return True
        return False
    #better to make this generic instead of only self.board
    def insert(self, board, column, mark):
        c = board[:,column]
        #search from bottom to top
        i = -1
        while c[i] != 0:
            i -=1
        c[i] = mark
        
         
    def insert_human(self, column, mark):
        
        while(not self.valid_choice(column)):
            print("please enter a valid choice")
            column = input()
            column = int(column)
        if self.valid_choice(column):
            c = self.board[:,column]
            #search from bottom to top
            i = -1
            while c[i] != 0:
                i -=1
            c[i] = mark
            
        
    def winning_move(self, mark):
        #Horizontal
        for c in range(COLUMNS-3):
            for r in range(ROWS):
                #use set operation to find winning combination
                #uncommenting search_space prints out the space being searched on the board
                search_space = np.zeros((6,7))
                search_space[r,c:c+4] = '1'
                #print(search_space)
                if list((set(self.board[r,c:c+4]))) == [mark]:
                    return True
        #Vertical
        for c in range(COLUMNS):
            for r in range(ROWS -3):
                #use set operation to find winning match
                search_space = np.zeros((6,7))
                search_space[r:r+4,c] = '1'
               # print(serach_space)
                if list(set(self.board[r:r+4,c])) == [mark]:
                    return True
                
             
        #Downwards Diag
        for c in range(COLUMNS-3):
            for r in range(ROWS -3):
                search_space = np.zeros((6,7))
                #combine coordinates
                #here r and c are the offsets
                i = [r + a for a in range(COLUMNS-3)]
                j = [c + b for b in range(COLUMNS-3)]
                
                coord = [i,j]
                search_space[coord[0],coord[1]] ='1'
               # print(search_space)
                if list(set(self.board[coord[0],coord[1]])) == mark:
                    return True
               
        #Upwards Diag
        for c in range(COLUMNS -3):
            for r in range(3, ROWS):
                search_space = np.zeros((6,7)) 
                #combine coordinates
                #here r and c are the offsets
                i = [r - a for a in range(COLUMNS-3)]
                j = [c + b for b in range(COLUMNS-3)]
                coord = [i,j]
                search_space[coord[0],coord[1]] ='1'
               
               # print(search_space)
                if list(set(self.board[coord[0],coord[1]])) == mark:
                    return True
                
        return False
    #error sometimes occurs in full column?
    def getLegal(self):
        legal = []
        if not self.available_space():
            return False
        else:
            for c in range(len(self.board[1])):
                #print(c)
                #numpy.where should return None if there is nothing
                if np.where(np.array(self.board[:,c])== 0):
                    blanks = np.where(np.array(self.board[:,c])== 0)
                    #skip if there is nothing in it         
                    if len(blanks[0]) == 0:
                        pass
                           
                    else:
                        valid = [np.max(blanks), c]
                        legal.append(valid)     
                     
                       
                        
                 
        return legal
    
    def humanPlayer(self, mark, isActive = False):
        val = input(f"player {mark}'s turn, enter column: \n")
        g.insert_human(int(val),mark)
        
    def randPlayer(self, mark):
        #get legal values and insert
        options = self.getLegal()
        #only need the column
        row,col = random.choice(options)
        self.insert(self.board, col, mark)
        #don't have to worry about invalid input
        
    def aiPlayer(self, mark):
        options = self.getLegal()
        duplicate = copy.deepcopy(self.board)
        scores = []
        for legal_move in options:
            #you only need the column values from self.getLegal()
            #to do an insert
            row, col = legal_move
            self.insert(duplicate, col,mark)
            #replace 2 with negative 1
            duplicate = np.where(duplicate == 2, -1, duplicate)
            #Need to do a whole bunch of conversions to get from
            #6x7 to size 42 tensor to dtype float
            score = self.model(torch.from_numpy(duplicate).flatten().float())
            scores.append(score)
            duplicate = copy.deepcopy(self.board) 
        #can be index of minimum or maximum value depending on who goes first
        min_val = min(scores)
        min_pos = scores.index(min_val)
        prediction = options[min_pos]
        row, col = prediction
        self.insert(self.board,col,mark)
           
        # empty = self.getEmpty()
        # board = board.astype(int)
        # duplicate = copy.deepcopy(board)
        # scores = []
        # for legal_moves in empty:
        #     #drop a -1 player 'O' at one of these valid locations
        #     #and predict the score, more positive is better
        #     #doing this weird conversion where you get positions on
        #     #the 3 by 3 square to index of one hot encoded vector, ex: [0, 0, 0, 0, 1]
        #     duplicate[0, int(legal_moves[0]*3) + int(legal_moves[1]) ] = '-1'
        #     score = classifier.predict(duplicate)
        #     #best practice is to associate score with a position on the board
        #     scores.append(score)
        #     #equivalent to reset board
        #     duplicate = copy.deepcopy(board)
        # #find maximum value of list
        # #the index of min val is the prediction
        # min_val = min(scores)
        # min_pos = scores.index(min_val)
        # prediction = empty[min_pos]
        # return prediction
    
               
                
        
                
               
                
            
                                                              
if __name__ == '__main__':
    
    
    
    
    
    
    g = Game()
    # g.board = np.array([
    # [0, 0, 0, 2, 0, 0, 0],
    # [0, 1, 0, 2, 0, 0, 1],
    # [0, 2, 0, 2, 0, 2, 2],
    # [0, 1, 2, 1, 1, 1, 2],
    # [0, 1, 1, 2, 2, 1, 1],
    # [0, 2, 2, 1, 1, 2, 1],
    # ])
    # valid = g.getLegal()
  

    print("done loading")
    mark = FIRST
    while ( True):
        print('\n')
        g.print_board()
        # #simulated
        # g.randPlayer( mark)
        #evaluate winning move 
        if (g.winning_move(mark)):
            print('\n')
            g.print_board()
           
            print(f"player {mark} has won the game!")
            break
        if mark == FIRST:
            # #call the AI player second
            g.humanPlayer(FIRST, True)
            # #g.print_board()
            mark = SECOND
           
        else:
            g.aiPlayer(mark)
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
 
            
        
       
            
