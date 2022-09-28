# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:36:28 2022

@author: LuTimothy
"""

import os
import re
import numpy as np
import random

#custom exception
class game_over(Exception):
    pass

COLUMNS = 7
ROWS = 6
FIRST = 1
SECOND = 2


        


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
    
    def getLegal(self):
        legal = []
        if not self.available_space():
            return False
        else:
            for c in range(len(self.board[1])):
                print(c)
                #numpy.where should return None if there is nothing
                if np.where(np.array(self.board[:,c])== 0):
                    blanks = np.where(np.array(self.board[:,c])== 0)
                    valid = [np.max(blanks), c]
                    legal.append(valid)
                    #skip over if none object is found in np.where
                else:
                    pass
        return legal
    
    def humanPlayer(self, mark, isActive = False):
        val = input(f"player {mark}'s turn, enter column: \n")
        g.insert_human(int(val),mark)
        
    def randPlayer(self, mark):
        #get legal values and insert
        options = self.getLegal()
        #only need the column
        row,col = random.choice(options)
        self.insert(col, mark)
        #don't have to worry about invalid input
        pass
        
                    
               
                
        
                
               
                
            
                                                              
if __name__ == '__main__':
    g = Game()
    valid = g.getLegal()
    mark = FIRST
    #GetData()
   ## load()
    print("done loading")
    
    while ( True):
        g.print_board()
        g.randPlayer( mark)
        #evaluate winning move 
        if (g.winning_move(mark)):
            g.print_board()
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
 
            
        
       
            
