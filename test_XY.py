# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:38:17 2021

@author: Giorgi
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import load

n_batches = load('data/n_batches.joblib')
X = load('data/X%d.joblib' % (n_batches-1))
Y = load('data/Y%d.joblib' % (n_batches-1))
model = load('models/q%d.joblib' % (n_batches-1))



j_plot = np.random.randint(0, len(Y), 100)

plt.scatter(Y[j_plot], model.predict(X)[j_plot], s=1)

plt.figure()
plt.scatter(X[j_plot, 0], Y[j_plot], s=1)
plt.scatter(X[j_plot, 0], model.predict(X)[j_plot], s=1)

plt.figure()
plt.scatter(X[j_plot, 1], Y[j_plot], s=1)
plt.scatter(X[j_plot, 1], model.predict(X)[j_plot], s=1)

plt.figure()
plt.scatter(X[j_plot, 2], Y[j_plot], s=1)
plt.scatter(X[j_plot, 2], model.predict(X)[j_plot], s=1)
