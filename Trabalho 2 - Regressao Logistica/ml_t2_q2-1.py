# PAULO BRUNO DE SOUSA SERAFIM      388149


import numpy as np
import matplotlib.pyplot as plt


# importacao de dados

with open('ex2data2.txt') as txt:
    A = [[float(x) for x in line.split(',')] for line in txt]
txt.close()


# configuracao das matrizes de treinamento e teste

X = np.array(A)
ncols = len(X[0]) - 1

Z = X[:, 2]
Y = X[:, 1]
X = X[:, 0]

plt.scatter(X, Y, c=Z, s = 80)
plt.show()
