#!/usr/bin/env python3
# coding: UTF-8

from sklearn.neural_network import MLPClassifier 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))

clf = MLPClassifier(random_state=42)
clf.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train))) 
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
