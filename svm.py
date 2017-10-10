#!/usr/bin/env python3
# coding: UTF-8

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

data = load_breast_cancer()
X = data.data
y = data.target

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale  = scaler.transform(X_test)

clf = SVC()
clf.kernel = 'linear'
clf.fit(X_train_scale, y_train)
print(clf.kernel)
print('Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}\n'.format(clf.score(X_test_scale, y_test)))

clf.kernel = 'rbf'
clf.fit(X_train_scale, y_train)
print(clf.kernel)
print('Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}\n'.format(clf.score(X_test_scale, y_test)))

clf.kernel = 'poly'
clf.fit(X_train_scale, y_train)
print(clf.kernel)
print('Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}\n'.format(clf.score(X_test_scale, y_test)))
