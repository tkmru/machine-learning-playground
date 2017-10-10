#!/usr/bin/env python3
# coding: UTF-8

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5.0, 5.0, 0.1)

plt.plot(x, np.tanh(x), label='tanh')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.show()
