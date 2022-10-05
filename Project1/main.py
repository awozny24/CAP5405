import classifiers
import regressors
import os
from sys import platform


path = os.getcwd() 

if platform == 'darwin':
    slash = '/'
else: 
    slash = '\\'

print("\n##################################################################")
print("############# BEGIN REGRESSOR CLASSIFICATION RESULTS #############")
print("##################################################################")
print()
print("#################################################")
print("############# Training on All Data ##############")
print("#################################################")
regressors.GetRegModel("lr", k_folds=10, print_acc=True, retrain=True)
regressors.GetRegModel("knn", k_folds=10, print_acc=True, retrain=True)
regressors.GetRegModel("mlp", k_folds=10, print_acc=True, retrain=True)
print()
print("#################################################")
print("############# Training on 1/10 Data #############")
print("#################################################")
regressors.GetRegModel("lr", k_folds=10, print_acc=True, retrain=True, dataProp=0.1)
regressors.GetRegModel("knn", k_folds=10, print_acc=True, retrain=True, dataProp=0.1)
regressors.GetRegModel("mlp", k_folds=10, print_acc=True, retrain=True, dataProp=0.1)
print("############## END REGRESSOR CLASSIFICATION RESULTS ################\n")

classifiers.knn('single', dataProp=1.0)
classifiers.knn('single', dataProp=0.1)
