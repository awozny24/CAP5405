# -*- coding: utf-8 -*-
"""GameDefinitions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P-If7swee1IE5dsTEiCimn8VinbzAm2V
"""

import random
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import os
import numpy as np
import classifiers

path = os.getcwd()

#list out data files
single = np.loadtxt((path +'\\tictac_single.txt'))




#resource 
#https://www.kaggle.com/code/ikarus777/tic-tac-toe/notebook

class gameLayout:

  def __init__(self):
    self.board = self.reset_board()
    self.mark = None
    self.winner = None
    pass
  
  def print_board(self):
    b = self.board
    print('     |     |')
    print('  ' + b[0][0] + '  |  ' + b[0][1] + '  |  ' + b[0][2])
    print('     |     |')
    print('-----------------')
    print('     |     |')
    print('  ' + b[1][0] + '  |  ' + b[1][1] + '  |  ' + b[1][2])
    print('     |     |')
    print('-----------------')
    print('     |     |')
    print('  ' + b[2][0] + '  |  ' + b[2][1] + '  |  ' + b[2][2])
    print('     |     |') 
    pass

  def reset_board(self):
    self.board = [
            ['_', '_', '_'],
            ['_', '_', '_'],
            ['_', '_', '_']
        ]
    return self.board

  def getEmpty(self):
    empty = []
    for (i,row) in enumerate(self.board):
        for (j,value) in enumerate(row):
          if value == '_':
            empty.append([i,j])
  #  print(empty)
    return empty

  def randMove(self):
    empty = self.getEmpty()
    return random.choice(empty)

  def playMove(self, move, mark = None):
    row, col = move
    if self.board[row][col] != '_':
      return -1
    else:
      self.board[row][col] = mark
      #flip mark from X to O or from O to X for next player
      if mark == 'X':
        self.mark = 'O'
      else:
        self.mark = 'X'
      return 0

#TODO add condition to check which exact player has won
  def checkVictory(self):
    board = self.board
    if board[0][0]==board[0][1]==board[0][2]!='_':
      #set winner to whichever player satisfies victory condition
       self.winner = board[0][0]
       return True
    if board[1][0]==board[1][1]==board[1][2]!='_':
       self.winner = board[1][0]
       return True
    if board[2][0]==board[2][1]==board[2][2]!='_':
       self.winner = board[2][0]
       return True
    if board[0][0]==board[1][0]==board[2][0]!='_':
       self.winner = board[0][0]
       return True
    if board[0][1]==board[1][1]==board[2][1]!='_':
       self.winner = board[0][1]
       return True
    if board[0][2]==board[1][2]==board[2][2]!='_':
      self.winner = board[0][2]
      return True
    if board[0][0]==board[1][1]==board[2][2]!='_':
      self.winner = board[0][0]
      return True
    if board[0][2]==board[1][1]==board[2][0]!='_':
      self.winner = board[0][2]
      return True
    self.winner = None    
    return False
    
  def gameplay(self, mark = 'X', startingMove = True):
      if startingMove:
        move = self.randMove()
        self.playMove(move, mark)
        win = None
        self.print_board()
        #first game move after starting move

        while len(self.getEmpty()) != 0 and not win:
          #self.print_board()
          mark = self.mark
          move = self.randMove()
          self.playMove(move, mark)
          win = self.checkVictory()
          print('\n\n')
          self.print_board()
        
        if (win):
          print(f"Winner! is {mark}")
          self.print_board()
     
        pass



  def predict_move(self, classifier, board):
      prediction = None
      board = board.astype(int)
      empty = self.getEmpty()
      #for some reason this is making invalid predictions
      
      val = classifier.predict(board)
      prediction = [int(val//3), int(val%3)]
      #forced to use a random valid move if prediction is not valid
      if (not prediction in empty):
          return random.choice(empty) 
     
  
      return prediction
  
    
  def gameplay_classification(self, classifier = None, mark = 'X', startingMove = True):
    if startingMove:
        move = self.randMove()
        self.playMove(move, mark)
        win = None
        self.print_board()
        #first game move after starting move

        while len(self.getEmpty()) != 0 and not win:
          #self.print_board()
          #create a copy of the board
          #for now pretend you didn't have to remove '_' char
          mark = self.mark
          
          #note only use predicted moves from classifier if it
          #is the computers turn otherwise use random move
          if (mark == 'O'):
              #prepare gameboard for classification
              test_board = self.board.copy()
              #replace characters for test_board using lambda
              for index, row in enumerate(test_board):
                  test_board[index] = list(map(lambda x: x.replace('X', '1'), test_board[index]))
                  test_board[index] = list(map(lambda x: x.replace('O', '-1'), test_board[index]))
                  test_board[index] = list(map(lambda x: x.replace('_', '0'), test_board[index]))

              test_board = np.array(test_board)
              test_board = test_board.flatten()
              test_board = test_board.reshape(1,-1)
              move = self.predict_move(classifier,test_board)
    
              #use SVM regression
              #move = self.randMove()
              self.playMove(move, mark)
              win = self.checkVictory()
              print('\n\n')
              self.print_board()
              
          elif(mark == 'X'):
              move = self.randMove()
              self.playMove(move, mark)
              win = self.checkVictory()
              print('\n\n')
              self.print_board()
               
        
        if (win):
          print(f"Winner! is {mark}")
          self.print_board()
     
        pass
   


layout = gameLayout()
layout.print_board()


randchoices = layout.getEmpty()

'''
no learning involved just random choices
player wins by blind luck
'''
layout.gameplay()

''' 
uses classifiers
trained on single dataset
to predict the best move for 
player O
player X still makes 
random moves

more often than not player O should win
'''
# svm = classifiers.svm('single')
# print("playing game trained on single dataset using linear svm classifier")
# layout.reset_board()
# layout.gameplay_classification(classifier = svm)


# mlp = classifiers.mlp('single')
# print("playing game trained on single dataset using mlp classifier")
# layout.reset_board()
# layout.gameplay_classification(classifier = mlp)

knn = classifiers.knn('single')
print("playing game trained on single dataset using linear svm classifier")
layout.reset_board()
layout.gameplay_classification(classifier = knn)

