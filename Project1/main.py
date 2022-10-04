import numpy as np
arr = np.asarray(range(0, 9)).reshape(3, 3)
arr[0, 1] = 10
print(arr)

maxi = np.argmax(np.amax(arr, axis=1))
print(maxi)

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



# # IN CONNECT4_2.PY different tries
# elif (self.dlFrameWork == 'tensorflow') | (self.dlFrameWork == 'TensorFlow') | (self.dlFrameWork == 'Tensorflow'): 
#                 # print(duplicate.flatten())
#                 score = self.model.predict(duplicate.flatten().reshape(1, duplicate.shape[0]*duplicate.shape[1]))
#                 drawing = score[0][0]
#                 losing = score[0][1]
#                 winning = score[0][2]
#                 # score = winning*losing + drawing/2 * winning
#                 score = winning * (1 - losing) + drawing * winning
#                 storeStr.append(str(drawing) + '+' + str(losing) + '+' + str(winning))
#                 store.append([drawing, losing, winning])
#                 # score = winning * losing * drawing
#                 # score = winning * losing

#                 # currBoard = copy.deepcopy(self.board)
#                 # currBoard[row, col] = 1
#                 # duplicate2 = copy.deepcopy(currBoard)
#                 # options2 = self.getLegal()
#                 # for it2, o in enumerate(options2):
#                 #     duplicate2 = copy.deepcopy(currBoard)
#                 #     duplicate2[o[0], o[1]] = 1
#                 #     duplicate2 = np.where(duplicate2 == 2, -1, duplicate2)
#                 #     score2 = self.model.predict(duplicate2.flatten().reshape(1, duplicate2.shape[0]*duplicate2.shape[1]))
#                 #     # prob_mat[it1, it2] = score2[0][2] * score2[0][1] + score2[0][0]/2
#                 #     prob_mat[it1, it2] = score2[0][1]* score2[0][1]

#                 # score = score2[0][2]
#                 # score = self.model.predict(duplicate.flatten().reshape(1, duplicate.shape[0]*duplicate.shape[1]))
#                 # score = score[0][2]
                    
#             scores.append(score)
#             duplicate = copy.deepcopy(self.board) 
#         print(storeStr)
#         #can be index of minimum or maximum value depending on who goes first
#         if (self.dlFrameWork == 'pytorch') | (self.dlFrameWork == 'Pytorch'):
#             min_val = min(scores)
#             min_pos = scores.index(min_val)
#             prediction = options[min_pos]
#             row, col = prediction
#         elif (self.dlFrameWork == 'tensorflow') | (self.dlFrameWork == 'TensorFlow') | (self.dlFrameWork == 'Tensorflow'): 
#             # max_val = max(scores)
#             # max_pos = scores.index(max_val)
#             # prediction = options[max_pos]
#             # row, col = prediction
#             # prob_mat = scores * np.ones(prob_mat.shape) * prob_mat
#             # prob_mat = scores + prob_mat
#             # prob_mat = scores * prob_mat
#             # col = np.argmax(np.amax(prob_mat, axis=1), axis=0)
#             # row, col = np.where(prob_mat == np.amax(prob_mat))
#             # print(prob_mat)

#             # find the 3 smallest loss values 
#             store = np.asarray(store)
#             ind_sort_loss = np.argsort(store[:, 1])
#             first3 = ind_sort_loss[0:3]

#             # set draw values to 0
#             store[ind_sort_loss[3:], 0] = 0

#             # find 3 largest draw values
#             # ind_sort_draw = np.argsort(store[:, 0])
#             # last3 = ind_sort_draw[-3:]
#             # store[ind_sort_draw[:-3], 2] = 0

#             print(store)

#             store2 = store[:, 0]/2 + store[:, 1] * store[:, 2]

#             # find the index of the largest winning value
#             col = np.argmax(np.amax(store2))