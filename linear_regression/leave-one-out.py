#!/usr/bin/env python3
# coding: UTF-8

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()

x = data.data
y = data.target

loocv = LeaveOneOut()
train_index, test_index = next(loocv.split(x, y))
clf = LogisticRegression()
ave_score = cross_val_score(clf, x, y, cv=loocv)

print('{0:4.2f} +/- {1:4.2f} %'.format(ave_score.mean() * 100, ave_score.std() * 100))
