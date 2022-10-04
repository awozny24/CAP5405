import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier


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


def GetData(path):
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
    data[np.where(data=='loss')] = 1
    data[np.where(data=='win')] = 2
    

    data = data.astype(np.float)

    # return data
    return data[:, :-1], data[:, -1]

def RearrangeArray(X):
    if X.ndim != 1:
        numRows = X.shape[0]
    else:
        numRows = 1

    for row in range(0, numRows):
        if X.ndim != 1:
            x = X[row, :]
        else:
            x = X
        board = np.zeros([6, 7])
        for i in range(1, 8):
            start = 6*(i-1)
            end = 6*i
            step = 1
            board[:, i-1] = np.flip(x[start:end:step])
        if X.ndim != 1:
            X[row, :] = board.reshape(board.shape[0]*board.shape[1])
        else: 
            # print(board)
            X = board.reshape(board.shape[0]*board.shape[1])

    return X


def GetModel():
    # retrieve the data
    npC4DataPath = '.' + slash + "C4Data"
    XPath = npC4DataPath + '_X.npy'
    yPath = npC4DataPath + '_y.npy'
    if not (os.path.exists(XPath) and os.path.exists(yPath)):
        X, y = GetData(path)
        X = RearrangeArray(X)
        np.save(XPath, X)
        np.save(yPath, y)
    else:
        X = np.load(XPath)
        X = RearrangeArray(X)
        y = np.load(yPath)

    # split data in training, validation, and testing
    train_X, dummy_X, train_y, dummy_y = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=randomStateTF)# (, stratify=[-1, 0, 1])
    val_X, test_X, val_y, test_y = train_test_split(dummy_X, dummy_y, test_size=0.5, shuffle=True, random_state=randomStateTF)


    # mlpClass = MLPClassifier()
    # mlpClass.fit(train_X, train_y)
    # print(mlpClass.score(test_X, test_y))

    # convert y to one hot encoded
    train_y_onehot = to_categorical(train_y, num_classes=3)
    val_y_onehot = to_categorical(val_y, num_classes=3)
    test_y_onehot = to_categorical(test_y, num_classes=3)


    # max number of epochs
    max_epochs = 25
    batch_size = 500

    modelPath = "c4NN_modelRearranged"
    if not os.path.exists(modelPath):
        # initialize model and add layers
        c4NN = keras.models.Sequential(name="Connect4Model1")
        c4NN.add(keras.Input(shape=(X.shape[1],), sparse=False)) 
        c4NN.add(keras.layers.Dense(43, activation='relu', use_bias=True))
        c4NN.add(keras.layers.Dense(30, activation='relu', use_bias=True))
        c4NN.add(keras.layers.Dense(20, activation='relu', use_bias=True))
        c4NN.add(keras.layers.Dense(9, activation='relu', use_bias=True))
        c4NN.add(keras.layers.Dense(3, activation='softmax', use_bias=True, name="output"))

        # optimizer
        opt = keras.optimizers.Adam(learning_rate=0.01)

        # compile model
        c4NN.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        c4NN.summary()

        # train model
        modelHistory = c4NN.fit(train_X, train_y_onehot, 
                                validation_data=(val_X, val_y_onehot), 
                                epochs=max_epochs, 
                                batch_size=batch_size, 
                                shuffle=True)

        loss, acc = c4NN.evaluate(test_X, test_y_onehot, verbose=2)
        print("Model Accuracy: {:5.2f}%".format(100 * acc))
        c4NN.save(modelPath)

    else:
        c4NN = keras.models.load_model(modelPath)
        c4NN.evaluate(test_X, test_y_onehot, verbose=2)


    print(c4NN.predict(train_X[0].reshape(1, train_X.shape[1])))


GetModel()

