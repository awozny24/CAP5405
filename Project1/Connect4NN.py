import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from sys import platform
import os

if platform == 'darwin':
    slash = '/'
else: 
    slash = '\\'

path = os.getcwd()


# set seed
randomStateTF = 30
tf.random.set_seed(randomStateTF)


def GetData(path)
    # get Connect4 data as a list
    data = []
    with open(path + slash + "connectfour.data", "rt") as infile:
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

    return data


# create model
# c4NN = keras.models.Sequential(name="Connect 4 Neural Network")
# c4NN.Input(shape="")