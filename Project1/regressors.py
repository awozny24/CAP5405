import numpy as np
import os
from sys import platform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import shuffle


if platform == 'darwin':
    slash = '/'
else: 
    slash = '\\'

path = os.getcwd() 

randomState = 30

# set seed
np.random.seed(randomState)




########### Begin Linear Regression Model ##########
class LR:

    # initialize model and parameters
    def __init__(self):
        self.w = []
        self.bias = True


    # fit the model to the training data
    def fit(self, X, y, bias=True, alpha=0.1, max_num_iter=2500, thresh=0.000001):

        # set bias to user preference
        self.bias = bias

        # add bias vector to training X matrix if specified and reflect in shape of weights vectors
        if bias:
            trainX = np.append(X, np.ones((X.shape[0], 1)), 1)
        else:
            trainX = np.copy(X)

        # for each "class" (each possible space on tic tac toe)
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

        # convert list of weights to numpy array
        self.w = np.asarray(self.w)
        self.w = np.transpose(self.w)
        

    # make a prediction using the learned weights
    def predict(self, X, thresh=None, eliminateUsedSquares=True):
        # add bias vector to training X matrix if added 
        if self.bias:
            if (X.ndim == 1):
                testX = np.append(X, 1)
            else:
                testX = np.append(X, np.ones((X.shape[0], 1)), 1)
        else:
            testX = np.copy(X)

        # determine and return the prediction
        pred = testX.dot(self.w)
        if eliminateUsedSquares:
            pred = EliminateUsedSquares(pred, X)
        pred = GetRegressorPredictions(pred, thresh)
        return pred

######### End Linear Regression Model #########

######### Begin Regressor Wrapper #########
class Regressor:
    def __init__(self, model):
        self.model = model

    
    def fit(self, trainX, trainy):
        self.model.fit(trainX, trainy)


    def predict(self, X, thresh=None, eliminateUsedSquares=True):
        # make prediction and return it
        pred = self.model.predict(X)
        if eliminateUsedSquares:
            pred = EliminateUsedSquares(pred, X)
        pred = GetRegressorPredictions(pred, thresh)

        return pred


######### End Regressor Wrapper #########



######### Helper Functions ###########

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

        return trainX, testX, trainy, testy

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
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # initialize empty lists for storing folds
    listFoldTrain = []
    listFoldTest  = []
    
    # for each fold
    fold = 0
    for trainInd, testInd in kf.split(data):
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


def GetRegressorPredictions(pred, thresh=None):

    if thresh is None:
        # find the index of the next move from the maximum index of the linear regression predictions
        pred = np.argmax(pred, axis=1)
        return pred
    else:
        # set predictions above thresh to 1, and the ones below thresh to 0 and return
        pred[pred[:, np.array(range(0, pred.shape[1]))] >= thresh] = 1
        pred[pred[:, np.array(range(0, pred.shape[1]))] < thresh] = 0
        return pred


# set the probability to 0 if the position is already taken
def EliminateUsedSquares(pred, X):

    # find the location of the 0's and 1's (i.e. which spots are taken and which are not)
    zerosLoc = np.where(X == 0)
    onesLoc = np.where(X == 1)

    # switch 1's and 0's using XOR 
    arr = np.copy(X)
    arr[zerosLoc] = 1
    arr[onesLoc] = 0

    # multiply matrix of 1's and 0's
    # now the probabilities where the spots are taken are set to 0
    pred = np.multiply(pred, arr)
    return pred


# TODO: ADD TO THIS FUNCTIONS for Linear Regression
# TODO: CHECK THIS FUNCTION AND ABOVE FUNCTION
# TODO: How to specifically calculate accuracy
def PredictionAccuracy(pred, y):
    if pred.ndim == 1:
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


# Get Confusion Matrix
def GetConfusionMatrix(X, y, model):
    title = [("Normalized confusion matrix")]

    

    # GetRegressorPredictions

    # title = [("Normalized confusion matrix")]
    # disp = ConfusionMatrixDisplay.from_estimator(
    #     classifier,
    #     testX,
    #     testy,
        
    #     cmap=plt.cm.Blues,
    #     normalize= 'true',
    # )
    # disp.ax_.set_title(title)

    # print(title)
    # print(disp.confusion_matrix)
    # plt.show()


######### End Helper Functions ##########


# get data in folds
X_train_fold, X_test_fold, y_train_fold, y_test_fold = LoadMultiData(path, k_folds=5, random_state=randomState, shufflePostSplit=True)


runLR = True
runKNNR = True
runMLPR = True

# Linear Regression Training and Testing
# TODO: IMPLEMENT FOR KFOLD
# train and predict next moves
if (runLR):

    for i, (X_train, X_test, y_train, y_test) in enumerate(zip(X_train_fold, X_test_fold, y_train_fold, y_test_fold)):
        # Sklearn Linear Regression
        # predL, sklearnLRModels = TrainAndPredictNextMove(sklearn.linear_model.LinearRegression())
        # print(f"Number of Correct Predictions: {sum(y_test[np.array(range(0, y_test.shape[0])), predL])}")
        # print("Accuracy Sklearn Linear Regression: {:.2f}%".format(LRAccuracy(predL, y=y_test)*100))
        # print()

        # create and train Own Implementation of Linear Regression
        # lr = LR()
        lr = LR()
        lr.fit(X_train, y_train)

        # get predictions
        # 1 = valid move
        # 0 = invalid move
        predTrain = lr.predict(X_train, thresh=None, eliminateUsedSquares=True)
        predTest  = lr.predict(X_test, thresh=None, eliminateUsedSquares=True)

        # print accuracies
        print("Accuracy Linear Regression")
        print("  Training: {:.2f}%".format(PredictionAccuracy(predTrain, y=y_train)*100))
        print("  Testing: {:.2f}%".format(PredictionAccuracy(predTest, y=y_test)*100))
        print()



# K-Nearest Neighbors Training and Testing
if runKNNR:
    
    for i, (X_train, X_test, y_train, y_test) in enumerate(zip(X_train_fold, X_test_fold, y_train_fold, y_test_fold)):
        # create and train knn regressor model on training data
        # knnReg = Regressor(KNNR())
        knnReg = KNNR()

        knnReg.fit(X_train, y_train)

        # # get predictions
        # # 1 = valid move
        # # 0 = invalid move
        predTrain = knnReg.predict(X_train)
        predTrain = EliminateUsedSquares(predTrain, X_train)
        predTrain = GetRegressorPredictions(predTrain, thresh=0.5)
        predTest = knnReg.predict(X_test)
        predTest = EliminateUsedSquares(predTest, X_test)
        predTest = GetRegressorPredictions(predTest, thresh=0.5)

        # print results
        print("K-Nearest Neighbors Regressor Accuracy")
        print("  Training: {:.2f}%".format(PredictionAccuracy(predTrain, y_train)*100))
        print("  Testing: {:.2f}%".format(PredictionAccuracy(predTest, y_test)*100))
        print()


# Multi-Layer Perceptron Model Training and Testing
# create and train mlp regressor model on training data
if runMLPR:

    for i, (X_train, X_test, y_train, y_test) in enumerate(zip(X_train_fold, X_test_fold, y_train_fold, y_test_fold)):
        # define multilayer perceptron regressor model and fit to training data
        mlpReg = MLPRegressor()
        # mlpReg = Regressor(MLPRegressor())
        mlpReg.fit(X_train, y_train)

        # get predictions
        # 1 = valid move
        # 0 = invalid move
        predTrain = mlpReg.predict(X_train)
        predTrain = EliminateUsedSquares(predTrain, X_train)
        predTrain = GetRegressorPredictions(predTrain, thresh=0.5)
        predTest = mlpReg.predict(X_test)
        predTest = EliminateUsedSquares(predTest, X_test)
        predTest = GetRegressorPredictions(predTest, thresh=0.5)

        # print results
        print("MultiLayer Perceptron Regressor Accuracy")
        print("  Training: {:.2f}%".format(PredictionAccuracy(predTrain, y_train)*100))
        print("  Testing: {:.2f}%".format(PredictionAccuracy(predTest, y_test)*100))
        print()


        
