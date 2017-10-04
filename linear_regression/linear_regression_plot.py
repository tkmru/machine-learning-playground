#!/usr/bin/env python3
# coding: UTF-8

import pandas
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

x_value = df[['RM']].values
y_value = df[['MEDV']].values

model = LinearRegression()
model.fit(x_value, y_value)

plt.scatter(x_value, y_value, c='blue')
plt.plot(x_value, model.predict(x_value), color='red')
plt.xlabel('average room #')
plt.ylabel('Price MEDV')
plt.show()
