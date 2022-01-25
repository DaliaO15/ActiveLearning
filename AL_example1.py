from sklearn.datasets import load_iris
import pandas as pd
from sklearn.svm import SVC, LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imageio as io
import os

#save data information as variable
iris = load_iris()
data = pd.DataFrame(iris.data)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

#put target data into data frame
#0 is IrisSetosa, 1 is IrisVersicolour, 2 is Iris Virginica
target = pd.DataFrame(iris.target)#Lets rename the column so that we know that these values refer to the target values
target = target.rename(columns = {0: 'species'})

origdata = pd.concat([data, target], axis = 1)
#Check for null values
#print(origdata.isnull().sum()) #no missing values

#Working with only two columns
k1, k2 = 'petal_length', 'petal_width'
data_aux = origdata[[k1, k2, 'species']].copy()
X = data_aux[[k1, k2]]
y = data_aux['species']
# plt.figure()
# setosa = y == 0
# versicolor = y == 1
# virginica = y == 2
# plt.scatter(X[k1][versicolor], X[k2][versicolor], c='r')
# plt.scatter(X[k1][virginica], X[k2][virginica], c='c')
# plt.xlabel(k1)
# plt.ylabel(k2)
# #plt.show()

#We discard the samples of Iris-setosa.
X1 = X[y != 0].reset_index(drop=True)
y1 = y[y != 0].reset_index(drop=True)
y1 -= 1
print(X1[:5])
fig = plt.figure()

plt.scatter(X1[k1][y1==0], X1[k2][y1==0], c='r')
plt.scatter(X1[k1][y1==1], X1[k2][y1==1], c='c')

plt.xlabel(k1)
plt.ylabel(k2)
#fig.savefig('main.jpg', dpi=100)
plt.show()


