import numpy as np
import os
from sys import platform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold

if platform == 'darwin':
    slash = '/'
else: 
    slash = '\\'

path = os.getcwd() 

randomState = 30

# set seed
np.random.seed(randomState)


def LoadMultiData(path, k_folds=10, random_state=42, shufflePostSplit=True):
    # load data into numpy arrays
    data = np.loadtxt(path + slash + 'tictac_multi.txt')
    # X = data[:, 0:9]
    # y = data[:, 9:]

    # if a number of folds is specified for splitting
    if k_folds is not None:
        # split data into k folds
        train_folds, test_folds = GetKFolds(data, k_folds, random_state, shufflePostSplit)

        # extract X and y data for training and testing
        # X data is in columns 0 to 8
        # y data is in columns 9 to 17
        trainX = [train_folds[i][:, 0:9] for i in range(0, len(train_folds))]
        trainy = [train_folds[i][:, 9:] for i in range(0, len(train_folds))]

        testX = [test_folds[i][:, 0:9] for i in range(0, len(test_folds))]
        testy = [test_folds[i][:, 9:] for i in range(0, len(test_folds))]

        return trainX, trainy, testX, testy

    # otherwise return raw data as X and y
    else:
        # separate data into X and y
        # X data is in columns 0 to 8
        # y data is in columns 9 to 17
        X = data[:, 0:9]
        y = data[:, 9:]

        # get training and testing data
        trainX, testX, trainy, testy = train_test_split(X, y, random_state=randomState, shuffle=True)

        return trainX, testX, trainy, testy


# function to get numpy array of data split into k specified folds
def GetKFolds(data, k=5, random_state=None, shufflePostSplit=True):

    # sklearn KFold object
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    # initialize empty lists for storing folds
    listFoldTrain = []
    listFoldTest  = []
    
    # for each fold
    fold = 0
    for trainInd, testInd in kf.split():
        # extract training and test data from determined indices
        listFoldTrain.append(data[trainInd, :])
        listFoldTest.append(data[testInd, :])
        
        # shuffle datasets after split if specified
        if shufflePostSplit:
            listFoldTrain[fold] = shuffle(listFoldTrain[fold], random_state=random_state)
            listFoldTest[fold]  = shuffle(listFoldTest[fold], random_state=random_state)
                    
        # next fold
        fold = fold + 1
        

    # put list data into numpy arrays
    train_folds = np.array(listFoldTrain, dtype=list)
    test_folds  = np.array(listFoldTest, dtype=list)
    
    return train_folds, test_folds 


X_train, X_test, y_train, y_test = LoadMultiData(path, k_folds=None, random_state=randomState, shufflePostSplit=True)


########### Linear Regression Model ##########
class LR:

    def __init__(self):
        self.w = []
        self.bias = True


    def fit(self, X, y, bias=True, alpha=0.1, max_num_iter=2500, thresh=0.000001):

        # set bias to user preference
        self.bias = bias

        # add bias vector to training X matrix if specified and reflect in shape of weights vectors
        if bias:
            trainX = np.append(X, np.ones((X.shape[0], 1)), 1)

        #     self.w = np.empty((X.shape[1]+1, 1))
        # else:
        #     self.w = np.empty((X.shape[1], 1))

        for i in range(0, y.shape[1]):

            # randomly initialize weights
            w = np.random.randint(-10, 10, X.shape[1])

            # add bias to weights if specified
            if bias:
                w = np.append(w, 1)

            # perform mini-batch stochastic gradient descent
            batch_size = trainX.shape[0]
            iter = 0
            step = np.random.randint(100, 101, w.shape[0])
            gradient = 2 / batch_size * np.transpose(trainX).dot(np.transpose(trainX.dot(w) - y[:,i]))
            while ((iter < max_num_iter) & (np.linalg.norm(step) > thresh)):

                # calculate gradient
                pred = np.dot(trainX, w)
                error = pred - y[:,i]
                gradient = np.transpose(trainX)
                gradient = np.dot(gradient, error)
                gradient = 2 / batch_size * gradient

                # calculate step size
                step = alpha * gradient

                # update weights
                w = w - step

                # increase iteration
                iter = iter + 1

            # w = np.reshape(w, (w.shape[0], 1))
            w_list = w.tolist()

            self.w.append(w_list)

            # # add weights vector to list of weights
            # self.w = np.append(self.w, w, axis=1)
        # self.w.append(np.ones(len(w_list)))

        # convert list of weights to numpy array
        self.w = np.asarray(self.w)
        self.w = np.transpose(self.w)
        

    def predict(self, X):
        # add bias vector to training X matrix if added 
        if self.bias:
            if (X.ndim == 1):
                testX = np.append(X, 1)
            else:
                testX = np.append(X, np.ones((X.shape[0], 1)), 1)

        return testX.dot(self.w)



def TrainAndPredictNextMove(regressorObj, testX=X_test, trainX=X_train, trainy=y_train):
    # train each model for each y vector
    reg = []
    mat = []
    for i in range(0, trainy.shape[1]):
        # append and train model for each y vector
        reg.append(regressorObj)
        reg[i].fit(trainX, trainy[:, i])

        # put predictions into list for later
        mat.append(reg[i].predict(testX))

    # put list of prediction vectors into tuple
    mat = tuple(mat)

    # determine the index maximum value of each models prediction for each sample
    # the index is the prediction of which tictactoe square to move to next
    stacked_mat = np.stack(mat)
    pred = np.argmax(stacked_mat, axis=0)

    return pred, reg


def LRAccuracy(pred, y=y_test):

    # count the number of a valid move was predicted
    numValidMoves = sum(y_test[np.array(range(0, y_test.shape[0])), pred])

    # calculate the accuracy of the prediction
    acc = numValidMoves / y_test.shape[0]

    return acc

######### End Linear Regression Model #########

# TODO: CHECK np.argmax()
def GetRegressorPredictions(pred, model=None, thresh=0.5):
    if (model == None):
        print("Please insert model to test for accuracy!!")
    elif (model == 'LR') or (model=='lr') or (model=="Linear Regression") or (model=="linear regression"):
        # find the index of the next move from the maximum index of the linear regression predictions
        pred = np.argmax(pred, axis=1)
        return pred
    else:
        # set predictions above 0.5 to 1, and the ones below 0.5 to 0 and return
        pred[pred[:, np.array(range(0, pred.shape[1]))] >= thresh] = 1
        pred[pred[:, np.array(range(0, pred.shape[1]))] < thresh] = 0
        return pred


# TODO: ADD TO THIS FUNCTIONS for Linear Regression
# TODO: CHECK THIS FUNCTION AND ABOVE FUNCTION
# TODO: How to specifically calculate accuracy
def PredictionAccuracy(pred, y=y_test, model=None):
    if (model == None):
        print("Please insert model to test for accuracy!!")
    elif (model == 'LR') or (model=='lr') or (model=="Linear Regression") or (model=="linear regression"):
        # use pred to find whether the predicted index is a valid move
        row_indices = np.asarray([i for i in range(0, y.shape[0])])
        acc = y[[row_indices],[pred]] == np.ones(pred.shape[0])
        acc = sum(acc[0]) / y.shape[0]
        return acc
    else:
        # count the number of correct predictions and divide by the total number of predictions
        numCorrect = sum(sum(pred == y))
        numTotal = y.shape[0]*y.shape[1]  
        acc = numCorrect / numTotal
        return acc



runLR = True
runKNNR = True
runMLPR = True

# Linear Regression Training and Testing
# TODO: IMPLEMENT FOR KFOLD
# train and predict next moves
if (runLR):
    # Sklearn Linear Regression
    # predL, sklearnLRModels = TrainAndPredictNextMove(sklearn.linear_model.LinearRegression())
    # print(f"Number of Correct Predictions: {sum(y_test[np.array(range(0, y_test.shape[0])), predL])}")
    # print("Accuracy Sklearn Linear Regression: {:.2f}%".format(LRAccuracy(predL, y=y_test)*100))
    # print()

    # create and train Own Implementation of Linear Regression
    lr = LR()
    lr.fit(X_train, y_train)

    # get predictions
    # 1 = valid move
    # 0 = invalid move
    predTrain = lr.predict(X_train)
    predTrain = GetRegressorPredictions(predTrain, model="LR")
    predTest  = lr.predict(X_test)
    predTest = GetRegressorPredictions(predTest, model="LR")

    # print accuracies
    print("Accuracy Linear Regression")
    print("  Training: {:.2f}%".format(PredictionAccuracy(predTrain, y=y_train, model="LR")*100))
    print("  Testing: {:.2f}%".format(PredictionAccuracy(predTest, y=y_test, model="LR")*100))
    print()



# K-Nearest Neighbors Training and Testing
if runKNNR:
    
    # create and train knn regressor model on training data
    knnReg = KNNR()
    knnReg.fit(X_train, y_train)

    # get predictions
    # 1 = valid move
    # 0 = invalid move
    predTrain = knnReg.predict(X_train)
    predTrain = GetRegressorPredictions(predTrain, model="KNNR")
    predTest = knnReg.predict(X_test)
    predTest = GetRegressorPredictions(predTest, model="KNNR")

    # print results
    print("K-Nearest Neighbors Regressor Accuracy")
    print("  Training: {:.2f}%".format(PredictionAccuracy(predTrain, y_train, model="KNNR")*100))
    print("  Testing: {:.2f}%".format(PredictionAccuracy(predTest, y_test, model="KNNR")*100))
    print()


# Multi-Layer Perceptron Model Training and Testing
# create and train mlp regressor model on training data
if runMLPR:
    mlpReg = MLPRegressor()
    mlpReg.fit(X_train, y_train)

    # get predictions
    # 1 = valid move
    # 0 = invalid move
    predTrain = mlpReg.predict(X_train)
    predTrain = GetRegressorPredictions(predTrain, model="MLPR")
    predTest = mlpReg.predict(X_test)
    predTest = GetRegressorPredictions(predTest, model="MLPR")

    # print results
    print("MultiLayer Perceptron Regressor Accuracy")
    print("  Training: {:.2f}%".format(PredictionAccuracy(predTrain, y_train, model="MLPR")*100))
    print("  Testing: {:.2f}%".format(PredictionAccuracy(predTest, y_test, model="MLPR")*100))
    print()
    
