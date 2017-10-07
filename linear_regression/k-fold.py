#!/usr/bin/env python3
# coding: UTF-8

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

data = load_breast_cancer()

x = data.data
y = data.target

clf = LogisticRegression()

# K-fold cross-validation: the average of the values computed in the loop
s = StratifiedKFold(n_splits=10, shuffle=True)
ave_score = cross_val_score(clf, x, y, cv=10)

print("{0:4.2f} +/- {1:4.2f} %".format(ave_score.mean() * 100, ave_score.std() * 100))
