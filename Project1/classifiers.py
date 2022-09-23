# -*- coding: utf-8 -*-
"""Classifiers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ssvkNL1C2j3ncwvMpg8_tZV71Hh3t5c7
"""

#remember player X is +1 and player O is -1 empty squares are 0 when loading dataset!
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import os



path = os.getcwd() 

def load(path):
  text = np.loadtxt(path)
  X = text[:,:9]
  Y = text[:,9:]
  return X, Y


def mlp(type = None):
    if type == 'single':
        X,Y = load(path + '\\tictac_single.txt')
    elif type == 'final':
         X,Y = load(path + '\\tictac_final.txt')
         
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=40)
    mlp = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    
    #prediction = mlp.predict(X_test)
    
    return mlp
         

def knn(type = None):
    if type == 'single':
        X,Y = load(path + '\\tictac_single.txt')
    elif type == 'final':
         X,Y = load(path + '\\tictac_final.txt')
    
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(Y), random_state=40, test_size=5, shuffle=True)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    
    #predictions = neigh.predict(X_test)
    
    return neigh
    
    
    

#want to change this to just svm
def svm(type = None):
    if type == 'single':
        X,Y = load(path + '\\tictac_single.txt')
    elif type == 'final':
         X,Y = load(path + '\\tictac_final.txt')
       
    # Splitting training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(Y), random_state=40, test_size=5, shuffle=True)

    # Creating and fitting the model
    svm = SVC(kernel='linear', gamma= 'scale', random_state= 40)
    svm.fit(X_train, y_train)
    
    #accuracy = svm.score(X_test, y_test)
    
    #predictions = svm.predict(X_test)
 
    return svm


def tenfold_crossvalid(clf, X, Y):
    scores = cross_val_score(clf, X, Y, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    

    

'''
in your code you will have
this somewhere along for confusion matrix


  C = confusion_matrix(y_test, y_pred)
        C = C.astype(np.float) / C.astype(np.float).sum(axis=1)
        print("Confusion Matrix: ")

to read in data you will want to have something like this

    """ Function for loading dataset from a text file and slicing in features and labels"""
    content = np.loadtxt(file_name)
    X = content[:,:9]
    y = content[:,9:]
    return X,y

    why?

        '''

