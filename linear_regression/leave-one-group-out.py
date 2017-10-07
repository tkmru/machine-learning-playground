#!/usr/bin/env python3
# coding: UTF-8

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()

x = data.data
y = data.target

# data don't have group, group is made here
group = np.array(list(range(50))*12)
group = np.sort(group[:y.size])

loocv = LeaveOneGroupOut()
clf = LogisticRegression()

ave_score = cross_val_score(clf, x, y, groups=group, cv=loocv)

print('size: {0}'.format(ave_score.size))
print('{0:4.2f} +/- {1:4.2f} %'.format(ave_score.mean() * 100, ave_score.std() * 100))
