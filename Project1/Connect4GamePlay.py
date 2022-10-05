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
import pygame
import sys
import math
import tensorflow as tf
import tensorflow.keras as keras

#work on making pygame UI

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

BLUE = (0,0,255)
BLACK =  (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
YELLOW = (255,255,0)

SQUARESIZE = 100
 
        


class Game():
    def __init__ (self, dlFrameWork):
        self.board = np.zeros((ROWS,COLUMNS))
        self.dlFrameWork = dlFrameWork
        #initialize FFNN model here
        if (dlFrameWork == 'pytorch') | (dlFrameWork == 'Pytorch'):
            path = 'FFNN_MODEL.pt'
            model = torch.load(path)
            model.eval()
            self.model = model
        elif (dlFrameWork == 'tensorflow') | (dlFrameWork == 'TensorFlow') | (dlFrameWork == 'Tensorflow'): 
            path = 'c4NN_modelRearranged'
            model = keras.models.load_model(path)
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
                if list(set(self.board[coord[0],coord[1]])) == [mark]:
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
                if list(set(self.board[coord[0],coord[1]])) == [mark]:
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
    
    #rework human Player for UI 
    #no more command line
    def humanPlayer(self, mark, col):
        #val = input(f"player {mark}'s turn, enter column: \n")
        #g.insert_human(int(val),mark)
        
        #check if valid
        if self.valid_choice(col):
            self.insert(self.board, col, mark)
        #do nothing if not valid
        else:
            pass
        
      
        
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
        store = []
        for legal_move in options:
            #you only need the column values from self.getLegal()
            #to do an insert
            row, col = legal_move
            self.insert(duplicate, col,mark)
            #replace 2 with negative 1
            duplicate = np.where(duplicate == 2, -1, duplicate)
            #Need to do a whole bunch of conversions to get from
            #6x7 to size 42 tensor to dtype float
            if (self.dlFrameWork == 'pytorch') | (self.dlFrameWork == 'Pytorch'):
                score = self.model(torch.from_numpy(duplicate).flatten().float())

            elif (self.dlFrameWork == 'tensorflow') | (self.dlFrameWork == 'TensorFlow') | (self.dlFrameWork == 'Tensorflow'): 
                # print(duplicate.flatten())
                score = self.model.predict(duplicate.flatten().reshape(1, duplicate.shape[0]*duplicate.shape[1]))

                # get probabilities for draw, lose, win
                drawing = score[0][0]
                losing = score[0][1]
                winning = score[0][2]

                # combine probabilites to get the best score
                score = winning * losing

                # store history
                store.append([drawing, losing, winning, score])

            scores.append(score)
            duplicate = copy.deepcopy(self.board) 
        #can be index of minimum or maximum value depending on who goes first
        if (self.dlFrameWork == 'pytorch') | (self.dlFrameWork == 'Pytorch'):
            min_val = min(scores)
            min_pos = scores.index(min_val)
            prediction = options[min_pos]
            row, col = prediction
        elif (self.dlFrameWork == 'tensorflow') | (self.dlFrameWork == 'TensorFlow') | (self.dlFrameWork == 'Tensorflow'): 
            max_val = max(scores)
            max_pos = scores.index(max_val)
            prediction = options[max_pos]
            row, col = prediction

        self.insert(self.board,col,mark)
           
        
        

#pygame UI borrowed from askPython    
def draw_board(board):
    for c in range(COLUMNS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, WHITE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
     
    #copy board
    copy_board = copy.deepcopy(board)
    copy_board = np.flipud(copy_board)
    for c in reversed(range(COLUMNS)):
        for r in range(ROWS):      
            if copy_board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif copy_board[r][c] == 2: 
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    pygame.display.update()
               
  
                            
                
            
                                                              
if __name__ == '__main__':
    #initalize pygame
    pygame.init()

  
    #define our screen size
    SQUARESIZE = 100
  
    #define width and height of board
    width = COLUMNS * SQUARESIZE
    height = (ROWS+1) * SQUARESIZE
  
    size = (width, height)
  
    RADIUS = int(SQUARESIZE/2 - 5)
  
    screen = pygame.display.set_mode(size)
   
    
    g = Game('tensorflow')
    
    
    
    # valid = g.getLegal()
   
    #Calling function draw_board again
    draw_board(g.board)
    pygame.display.update()
  
    myfont = pygame.font.SysFont("monospace", 75)
    game_over = False
    
    

    print("done loading")
    mark = FIRST
    # while ( True):
    #     print('\n')
    #     g.print_board()
    #     # #simulated
    #     # g.randPlayer( mark)
    #     #evaluate winning move 
    #     if (g.winning_move(mark)):
    #         print('\n')
    #         g.print_board()
           
           
    #         print(f"player {mark} has won the game!")
    #         break
    #     if mark == FIRST:
    #         # #call the AI player second
            
    #         # #g.print_board()
    #         mark = SECOND
           
    #     else:
    #         g.aiPlayer(mark)
    #         mark = FIRST
    #     if(g.winning_move(mark) == False and g.available_space() == False):
    #         print("game ended in a draw")
    #         break
    '''
    /***************************************************************************************
  *    Connect4 UI
  *    Author: ASKPYTHON
  *    Availability: https://www.askpython.com/python/examples/connect-four-game
  *
  ***************************************************************************************/
    '''
        


    
	#work on a new while loop for the game starting here
    #print board will need to be replaced with something else
  
    
    #define the shape of the UI in pixels
    while not game_over:
     
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit()
                
            if g.winning_move(mark):
                label = myfont.render(f"Player {mark} wins!!", 1,BLACK )
                screen.blit(label, (40,10))
                game_over = True
               
     
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, WHITE, (0,0, width, SQUARESIZE))
                posx = event.pos[0]
                if mark == FIRST:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
                else: 
                    pass
                   
            pygame.display.update()
     
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, WHITE, (0,0, width, SQUARESIZE))
                #print(event.pos)
                # Ask for Player 1 Input
                if mark == FIRST:
                    posx = event.pos[0]
                    col = int(math.floor(posx/SQUARESIZE))
                    g.humanPlayer(mark, col)
                  
                    mark = SECOND
                    
                        
                else:
                    #don't need to record on screen events
                    g.aiPlayer(mark)
                    mark = FIRST
                    
            draw_board(g.board)
pygame.time.wait(3000)
pygame.quit()
     
                   
     
                        
     
     
                # # # Ask for Player 2 Input
                # else:               
                #     posx = event.pos[0]
                #     col = int(math.floor(posx/SQUARESIZE))
     
                #     if is_valid_location(board, col):
                #         row = get_next_open_row(board, col)
                #         drop_piece(board, row, col, 2)
     
                #         if winning_move(board, 2):
                #             label = myfont.render("Player 2 wins!!", 1, YELLOW)
                #             screen.blit(label, (40,10))
                #             game_over = True
     
                # print_board(board)
                # draw_board(self.board)
     
                # turn += 1
                # turn = turn % 2
     
                # if game_over:
                #     pygame.time.wait(3000)
  