import numpy as np
import os
from sys import platform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if platform == 'darwin':
    slash = '/'
else: 
    slash = '\\'

path = os.getcwd() 

randomState = 40

# set seed
np.random.seed(randomState)

def loadData(path):
    # load data into numpy array and return
    data = np.loadtxt(path + slash + 'tictac_multi.txt')
    X = data[:, 0:9]
    y = data[:, 9:]
    return X, y


X, y = loadData(path)

# get training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=randomState, shuffle=True)

########### Linear Regression Model ##########
class LR:

    def __init__(self):
        self.w = None
        self.bias = True


    def fitModel(self, X, y, bias=True, alpha=0.1, max_num_iter=10000, thresh=0.00001):
    
        self.bias = bias

        # randomly initialize weights
        self.w = np.random.randint(-10, 10, X.shape[1])

        # add bias if requested
        if bias:
            # add bias weight to weights vector
            self.w = np.append(self.w, 1)

            # add bias vector to training X matrix
            trainX = np.append(X, np.ones((X.shape[0], 1)), 1)


        # perform mini-batch stochastic gradient descent
        batch_size = trainX.shape[0]
        iter = 0
        step = np.random.randint(100, 101, self.w.shape[0])
        gradient = 2 / batch_size * np.transpose(trainX).dot(np.transpose(trainX.dot(self.w) - y))
        while ((iter < max_num_iter) & (np.linalg.norm(step) > thresh)):
            # calculate gradient
            pred = np.dot(trainX, self.w)
            error = pred - y
            gradient = np.transpose(trainX)
            gradient = np.dot(gradient, error)
            gradient = 2 / batch_size * gradient

            # calculate step size
            step = alpha * gradient

            # update weights
            self.w = self.w - step

            # increase iteration
            iter = iter + 1

        # print(iter)


    def predictModel(self, X):
        # add bias vector to training X matrix if added 
        if self.bias:
            trainX = np.append(X, np.ones((X.shape[0], 1)), 1)

        return trainX.dot(self.w)


    def scoreModel(self, X, y):
        # add bias vector to training X matrix if added 
        if self.bias:
            trainX = np.append(X, np.ones((X.shape[0], 1)), 1)

        # prediction of >= 0.5 means a positive classification prediction
        # prediction of < 0.5 means a negative classification prediction
        pred = trainX.dot(self.w)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        # accuracy score
        score = np.sum(pred == y) / y.shape[0]     
        
        return score



# train each model for each y vector
lr = []
mat = []
for i in range(0, y_train.shape[1]):
    # append and train model for each y vector
    lr.append(LinearRegression())
    lr[i].fit(X_train, y_train[:, i])

    # put y matrix into list for later
    mat.append(lr[i].predict(X_train))

# put list of y vectors into tuple
mat = tuple(mat)

# determine the maximum value of each row of vectors
stacked_mat = np.stack(mat)
pred = np.argmax(stacked_mat, axis=0)

######### End Linear Regression Model #########