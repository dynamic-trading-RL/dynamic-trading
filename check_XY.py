# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:46:00 2021

@author: Giorgi
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import load

X = load('data/X0.joblib')
Y = load('data/Y0.joblib')

model = load('models/q0.joblib')


j_plot = np.random.randint(0, len(Y), 100)

fig = plt.figure()
plt.scatter(X[j_plot, 0], Y[j_plot], s=1, color='b')
plt.scatter(X[j_plot, 0], model.predict(X[j_plot]), s=1, color='r')

fig = plt.figure()
plt.scatter(X[j_plot, 1], Y[j_plot], s=1, color='b')
plt.scatter(X[j_plot, 1], model.predict(X[j_plot]), s=1, color='r')

fig = plt.figure()
plt.scatter(X[j_plot, 2], Y[j_plot], s=1, color='b')
plt.scatter(X[j_plot, 2], model.predict(X[j_plot]), s=1, color='r')




x_coordinates = (0, 1)
x0 = np.linspace(X[j_plot, x_coordinates[0]].min(),
                 X[j_plot, x_coordinates[0]].max())
x1 = np.linspace(X[j_plot, x_coordinates[1]].min(),
                 X[j_plot, x_coordinates[1]].max())
x3 = np.zeros(len(x0))

xx0 , xx1, xx3 = np.meshgrid(x0, x1, x3)

yy = model.predict(np.c_[xx0.ravel(),
                         xx1.ravel(),
                         xx3.ravel()]).reshape(xx0.shape)

plt.figure()
plt.contourf(xx0[:, :, 0], xx1[:, :, 0], yy[:, :, 0])
plt.colorbar()
