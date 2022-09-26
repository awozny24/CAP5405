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
FIRST = 'X'
SECOND = 'O'


def load():
    path = os.getcwd()
    file = open(path + '//connectfour.data', mode = 'r')
    #probably want to use string split
    outcomes = 'win|lose|draw'
    for line in file:
        match = re.search(outcomes,line)
        board, state = re.split(outcomes,line)
        #convert comma separated string to list
        board = board.split(',')
        board = np.array(list(board)).reshape(7,6)
        #remove empty values from board
        board = [x for x in board if x]
        #reorient
        board = np.flipud(board.T)
        state = match.group()
        
        # = np.array(list(board)).reshape(6,7)
        print()

class Game():
    def __init__ (self):
         self.win = 4
         self.board = np.zeros((ROWS,COLUMNS))
         
    def valid_choice(self, col):
        if(col>=0 and col<=6):
            #check top of column
            if(self.board[0][col] == 0):
                return True
        return False
        
         
    def insert(self, column, mark):
        c = self.board[column]
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
                if list(set(self.board[r][c:c+4])) == [mark]:
                    return True
        #Vertical
        for c in range(COLUMNS):
            for r in range(ROWS -3):
                #use set operation to find winning match
                if list(set(self.board[r:r+4][c])) == [mark]:
                    return True
        #Upwards Diag
        for c in range(COLUMNS-3):
            for r in range(ROWS -3):
                if list(set(self.board[r:r+4][c:c+4])) == [mark]:
                    return True
        #Downwards Diag
        for c in range(COLUMNS -3):
            for r in range(3, ROWS):
                if list(set(self.board[r:r-4:-1][c:c+4])) == [mark]:
                    return True
                                                              
if __name__ == '__main__':
	g = Game()
	mark = 'X'
	while True:
		g.printBoard()
		row = input('{}\'s turn: '.format('Red' if turn == RED else 'Yellow'))
		g.insert(int(row), turn)
		turn = YELLOW if turn == RED else RED                
               
 
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
 
            
        
       
            
