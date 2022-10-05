# -*- coding: utf-8 -*-
"""Classifiers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ssvkNL1C2j3ncwvMpg8_tZV71Hh3t5c7
"""

#remember player X is +1 and player O is -1 empty squares are 0 when loading dataset!
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
import os
from sys import platform



path = os.getcwd() 

if platform == 'darwin':
    slash = '/'
else: 
    slash = '\\'

def load(path, dataProp=1.0):
  text = np.loadtxt(path)
  # randomly select portion of data
  perm = np.random.permutation(range(0, text.shape[0]))
  text = text[perm[0:int(dataProp*text.shape[0])], :]
  X = text[:,:9]
  Y = text[:,9:]
  return X, Y


def mlp(type = None, dataProp=1.0):
    if type == 'single':
        X,Y = load(path + slash + 'tictac_single.txt', dataProp)
    elif type == 'final':
         X,Y = load(path + slash + 'tictac_final.txt', dataProp)


    
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(Y), random_state=40)
    mlp = MLPClassifier(activation='relu', solver ='adam', max_iter=300).fit(X_train, y_train)
   
    
    print(" cross validation scores:")
    crossvalid(mlp, X, Y)
    
   
    get_cmatrix(mlp, X_test, y_test)
    kf = KFold(n_splits=10, random_state=10, shuffle=True)
    acc_score = []
    for train_index , test_index in kf.split(X):
        try:
            X_train , X_test = X[train_index,:],X[test_index,:]
            Y_train , Y_test = Y[train_index] , Y[test_index]
            mlp.fit(X_train,np.ravel(Y_train))
            pred_values = mlp.predict(X_test)
            acc = accuracy_score(pred_values , np.ravel(Y_test))
            acc_score.append(acc)
            
        except:
            X_train , X_test = X[train_index,:],X[test_index,:]

       

    avg_acc_score = sum(acc_score)/10
 
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))


    
    return mlp
         

def knn(type = None, dataProp=1.0):
    if type == 'single':
        X,Y = load(path + slash + 'tictac_single.txt', dataProp)
    elif type == 'final':
         X,Y = load(path + slash + 'tictac_final.txt', dataProp)
    
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(Y), random_state=40, shuffle=True)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
 
    
    
    print(" cross validation scores:")
    crossvalid(neigh, X, Y)
  
    
    get_cmatrix(neigh, X_test, y_test)
    
    kf = KFold(n_splits=10, random_state=10, shuffle=True)
    acc_score = []
    for train_index , test_index in kf.split(X):
        try:
            X_train , X_test = X[train_index,:],X[test_index,:]
            Y_train , Y_test = Y[train_index] , Y[test_index]
            neigh.fit(X_train,np.ravel(Y_train))
            pred_values = mlp.predict(X_test)
            acc = accuracy_score(pred_values , np.ravel(Y_test))
            acc_score.append(acc)
          
        except:
            X_train , X_test = X[train_index,:],X[test_index,:]


   
    avg_acc_score = sum(acc_score)/10

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

    
    
    return neigh
    
    
    

#want to change this to just svm
def svm(type = None, dataProp=1.0):
    if type == 'single':
        X,Y = load(path + slash + 'tictac_single.txt', dataProp)
    elif type == 'final':
         X,Y = load(path + slash + 'tictac_final.txt', dataProp)
       
    # Splitting training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(Y), random_state=40, shuffle=True)

    # Creating and fitting the model
    svm = SVC(kernel='linear', gamma= 'scale', random_state= 40)
    svm.fit(X_train, y_train)
    crossvalid(svm, X, Y)
    
 
    get_cmatrix(svm, X_test, y_test)
    
    kf = KFold(n_splits=10, random_state=10, shuffle=True)
    acc_score = []
    for train_index , test_index in kf.split(X):
        try:
            X_train , X_test = X[train_index,:],X[test_index,:]
            Y_train , Y_test = Y[train_index] , Y[test_index]
            svm.fit(X_train,np.ravel(Y_train))
            pred_values = mlp.predict(X_test)
            acc = accuracy_score(pred_values , np.ravel(Y_test))
            acc_score.append(acc)
            
        except:
            X_train , X_test = X[train_index,:],X[test_index,:]
       

    avg_acc_score = sum(acc_score)/10
 
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    
 
 
    return svm


def crossvalid(clf, X, Y):
   
    scores = cross_val_score(clf, X, np.ravel(Y), cv =5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    

    
def get_cmatrix(classifier, X_test, y_test):
    title = [("Normalized confusion matrix")]
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        
        cmap=plt.cm.Blues,
        normalize= 'true',
    )
    disp.ax_.set_title(title)

    print(title)
    # print(disp.confusion_matrix)
    print("\t\t\tPredicted")
    for i, row in enumerate(disp.confusion_matrix):
        if i == int(len(row)/2):
            print('True\t[', end='')
        else:
            print('\t[', end='')
        for col in row:
                print("{:.3f} ".format(col), end='')
        print(']')

    plt.show()
