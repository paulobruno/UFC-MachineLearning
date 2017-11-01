# PAULO BRUNO DE SOUSA SERAFIM      388149


import numpy as np
import matplotlib.pyplot as plt


# importacao de dados

with open('ex1data1.txt') as txt:
    A = [[float(x) for x in line.split(',')] for line in txt]
txt.close()


# configuracao das matrizes de treinamento e teste

X = np.array(A)
ncols = len(X[0]) - 1

Y = X[:, ncols] # Y eh a ultima coluna da matriz
X = X[:, 0]


plt.plot(X, Y, 'bo')
plt.show()
