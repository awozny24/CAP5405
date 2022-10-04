True
# import numpy as np
# # import tensorflow as tf
# # import tensorflow.keras as keras


# # from sys import platform
# # import os

# # if platform == 'darwin':
# #     slash = '/'
# # else: 
# #     slash = '\\'

# # path = os.getcwd()


# # # set seed
# # randomStateTF = 30
# # tf.random.set_seed(randomStateTF)


# # def GetData(path):
# #     # get Connect4 data as a list
# #     data = []
# #     with open(path + slash + "connectfour.data", "rt") as infile:
# #         data.append([line.strip().split(',') for line in infile.readlines()])

# #     # get data as numpy array
# #     data = np.asarray(data)
# #     data = data[0]

# #     data[np.where(data=='b')] = 0
# #     data[np.where(data=='x')] = 1
# #     data[np.where(data=='o')] = -1
# #     data[np.where(data=='draw')] = 0
# #     data[np.where(data=='loss')] = 1
# #     data[np.where(data=='win')] = 2
    

# #     data = data.astype(np.float)

# #     # return data
# #     return data[0, :-1], data[0, -1]

# # # # retrieve the data
# # npDat = "npDat"
# # if not (os.path.exists(npDat + "_X.npy") and os.path.exists(npDat + "_y.npy")):
# #   print("HELP")
# #   X, y = GetData(path)
# #   np.save(npDat + '_X', X)
# #   np.save(npDat + '_y', y)

# # else:
# #   print("NO HELP")
# #   X = np.load(npDat + '_X.npy')
# #   y = np.load(npDat + '_y.npy')

# X = np.asarray([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  
#                  0.,  0.,  0.,  0.,  0.,  1., -1.,  
#                  0.,  0.,  0.,  0.,  1., -1.,  1., 
#                  -1., 1., -1.,  0.,  0.,  0.,  0.,  
#                  0.,  0.,  0.,  0.,  0.,  0.,  0.,  
#                  0.,  0.,  0.,  0.,  0.,  0.,  0.])

# # X = np.asarray([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,
# #   1., -1.,  1., -1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
# #   0.,  0.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,
# #   1., -1.,  1., -1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
# #   0.,  0.,  0.,  0.,  0.,  0.]])

# X = np.asarray([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  0.,  0.,
#   1., -1.,  1., -1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#   0.,  0.,  0.,  0.,  0.,  0.])


# def RearrangeArray(X):
#   if X.ndim != 1:
#       numRows = X.shape[0]
#   else:
#       numRows = 1

#   for row in range(0, numRows):
#     if X.ndim != 1:
#       x = X[row, :]
#     else:
#       x = X
#     board = np.zeros([6, 7])
#     for i in range(1, 8):
#       start = 6*(i-1)
#       end = 6*i
#       step = 1
#       board[:, i-1] = np.flip(x[start:end:step])
#     if X.ndim != 1:
#       X[row, :] = board.reshape(board.shape[0]*board.shape[1])
#     else: 
#       # print(board)
#       X = board.reshape(board.shape[0]*board.shape[1])

#   return X

# X = RearrangeArray(X)
# print(X)
