# PAULO BRUNO DE SOUSA SERAFIM      388149


import numpy as np
import matplotlib.pyplot as plt


# importacao de dados
with open('ex1data2.txt') as txt:
    A = [[float(x) for x in line.split()] for line in txt]
txt.close()


# configuracao das matrizes de treinamento e teste
M = np.array(A) # converte array simples para array numpy de A

nrows = len(M)

onesCol = np.ones((nrows,1)) # cria coluna de um's
M = np.append(onesCol, M, axis=1) # adiciona coluna de um's

ncols = len(M[0])


# atribuicao matrizes
np.random.shuffle(M) # embaralha as linhas

Y = M[:, ncols-1] # Y eh a ultima coluna da matriz
X = M[:,:-1] # X sao todas as colunas ate a penultima

X = (X - np.mean(X)) / np.std(X) # normalizacao


# treinamento
Xt = X.transpose()

w = np.dot(Xt, X)
w = np.linalg.inv(w)
w = np.dot(w, Xt)
w = np.dot(w, Y)


# execucao
Ynew = np.dot(X, w)


print w

plt.plot(Y)
plt.plot(Ynew, 'r')
plt.show()
