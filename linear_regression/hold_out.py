#!/usr/bin/env python3
# coding: UTF-8

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

data = load_breast_cancer()
x = data.data
y = data.target

clf = LogisticRegression()

s = StratifiedShuffleSplit(n_splits=10, train_size=0.70, test_size=0.30, random_state=0)

# hold-out: split test data, train data
for train_index, test_index in s.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
