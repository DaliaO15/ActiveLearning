""" This is a replication of the AL example posted in the
"Towards data sciences" blog. The post Active Learning Tutorial
— Machine Learning with Python was accessed on January 25th, 2022,
and it is found in the following link:
https://towardsdatascience.com/active-learning-5b9d0955292d
"""

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.svm import SVC, LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imageio as io
import os

# **************************DATA PRE-PROCESSING

# Save data information as variable and then dataframes
iris = load_iris()
data = pd.DataFrame(iris.data)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# 0 is Iris Setosa, 1 is Iris Versicolour, 2 is Iris Virginica
target = pd.DataFrame(iris.target)
target = target.rename(columns={0: 'species'})

origdata = pd.concat([data, target], axis=1)
# Check for null values
# print(origdata.isnull().sum()) #no missing values

# Working with only two columns
k1, k2 = 'petal_length', 'petal_width'
data_aux = origdata[[k1, k2, 'species']].copy()
X = data_aux[[k1, k2]]
y = data_aux['species']
# We discard the samples of Iris-setosa.
X1 = X[y != 0].reset_index(drop=True)
y1 = y[y != 0].reset_index(drop=True)
y1 -= 1
# fig = plt.figure()
# plt.scatter(X1[k1][y1 == 0], X1[k2][y1 == 0], c='r')
# plt.scatter(X1[k1][y1 == 1], X1[k2][y1 == 1], c='c')
# plt.xlabel(k1)
# plt.ylabel(k2)
# fig.savefig('main.jpg', dpi=100)
# plt.show()

# *****************************TRAIN WITH THE WHOLE DATASET

# Train a SVM classifier
y1 = y1.astype(dtype=np.uint8)
clf0 = LinearSVC()
clf0.fit(X1, y1)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
print("Weights assigned to the features")
print(clf0.coef_)
print("Constants in decision function")
print(clf0.intercept_)

# PLot the line founded
xmin, xmax = X1[k1].min(), X1[k1].max()
ymin, ymax = X1[k2].min(), X1[k2].max()
stepx = (xmax - xmin) / 99
stepy = (ymax - ymin) / 99
a0, b0, c0 = clf0.coef_[0, 0], clf0.coef_[0, 1], clf0.intercept_
# Formula for reference: a*x + b*y + c = 0   =>  y = -(a*x + c)/b
lx0 = [xmin + stepx * i for i in range(100)]
ly0 = [-(a0 * lx0[i] + c0) / b0 for i in range(100)]
# plt.figure()
# plt.scatter(X1[k1][y1 == 0], X1[k2][y1 == 0], c='r')
# plt.scatter(X1[k1][y1 == 1], X1[k2][y1 == 1], c='c')
# plt.plot(lx0, ly0, c='m')
# plt.xlabel(k1)
# plt.ylabel(k2)
# plt.title("LinearSVC")
# plt.show()

# ****************************ACTIVE LEARNING

# we split the dataset into two parts — pool(80%) and test(20%).
X_pool, X_test, y_pool, y_test = train_test_split(X1, y1, test_size=0.2, random_state=20)
X_pool, X_test, y_pool, y_test = X_pool.reset_index(drop=True), X_test.reset_index(drop=True), y_pool.reset_index(
    drop=True), y_test.reset_index(drop=True)

# Apply the decision function of the SVM on two data points
print("Decision for only two values from the pool set")
print(clf0.decision_function(X_pool.iloc[6:8]))

# Basic sampling technique
def find_most_ambiguous(clf, unknown_indexes):
    ind = np.argmin(np.abs(
        list(clf0.decision_function(X_pool.iloc[unknown_indexes]))))
    return unknown_indexes[ind]

# For ploting
def plot_svm(clf, train_indexes, unknown_indexes, new_index=False, title=False, name=False):
    X_train = X_pool.iloc[train_indexes]
    y_train = y_pool.iloc[train_indexes]

    X_unk = X_pool.iloc[unknown_indexes]

    if new_index:
        X_new = X_pool.iloc[new_index]

    a, b, c = clf.coef_[0, 0], clf.coef_[0, 1], clf.intercept_  # Straight Line Formula

    lx = [xmin + stepx * i for i in range(100)]
    ly = [-(a * lx[i] + c) / b for i in range(100)]

    fig = plt.figure(figsize=(9, 6))
    # plt.scatter(x[k1][setosa], x[k2][setosa], c='r')
    plt.scatter(X_unk[k1], X_unk[k2], c='k', marker='.')  # Unknown values are black dots
    plt.scatter(X_train[k1][y_train == 0], X_train[k2][y_train == 0], c='r', marker='o')
    plt.scatter(X_train[k1][y_train == 1], X_train[k2][y_train == 1], c='c', marker='o')

    plt.plot(lx, ly, c='m')
    plt.plot(lx0, ly0, '--', c='g')  # Hyperplane founded with all the points

    if new_index:
        plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
        plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
        plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
        plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
        plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)

    if title:
        plt.title(title)
    if name:
        fig.set_size_inches((9, 6))
        plt.savefig(name, dpi=100)

    plt.xlabel(k1)
    plt.ylabel(k2)
    # plt.show()


# Divide the pool in train and unlabeled data
train_indexes = list(range(10))
unknown_indexes = list(range(10, 80))
X_train = X_pool.iloc[train_indexes]
y_train = y_pool.iloc[train_indexes]
# Train the training data, HA!
clf = LinearSVC()
clf.fit(X_train, y_train)
# Storing
folder = "rs1it5/"  # folder = "rs2it20/"
try:
    os.mkdir(folder)
except:
    pass

# ------>>>>> Active learning
filenames = []
title = "Beginning"
name = folder + "rs1it5_0a.jpg"
# name = folder + ("rs2it20_0a.jpg")
plot_svm(clf, train_indexes, unknown_indexes, False, title, name)
filenames.append(name)

n = find_most_ambiguous(clf, unknown_indexes)
unknown_indexes.remove(n)

title = "Iteration 0"
name = folder + "rs1it5_0b.jpg"
# name = folder + ("rs2it20_0b.jpg")
filenames.append(name)
plot_svm(clf, train_indexes, unknown_indexes, n, title, name)
t = []
# for i in range(5):
for i in range(5):
    train_indexes.append(n)
    X_train = X_pool.iloc[train_indexes]
    y_train = y_pool.iloc[train_indexes]
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    title, name = "Iteration " + str(i + 1), folder + ("rs1it5_%d.jpg" % (i + 1))
    # title, name = "Iteration " + str(i + 1), folder + ("rs2it20_%d.jpg" % (i + 1))

    n = find_most_ambiguous(clf, unknown_indexes)
    unknown_indexes.remove(n)
    plot_svm(clf, train_indexes, unknown_indexes, n, title, name)
    filenames.append(name)

images = []
for filename in filenames:
    images.append(io.imread(filename))
io.mimsave('rs1it5.gif', images, duration=1)
# io.mimsave('rs1it20.gif', images, duration = 1)
try:
    os.mkdir('rs1it5')
#    os.mkdir('rt2it20')
except:
    pass
os.listdir('rs1it5')
