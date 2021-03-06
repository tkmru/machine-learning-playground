#!/usr/bin/env python3
# coding: UTF-8

import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y, label='relu')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.show()
