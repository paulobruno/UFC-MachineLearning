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


# inicializacao treinamento
w = np.ones(ncols-1)

alpha = 0.01
age = 100

EQM = np.zeros(age)


# treinamento
for i in range(0, age):
    
    np.random.shuffle(M) # embaralha as linhas

    Y = M[:, ncols-1]
    X = M[:,:-1]

    X = (X - np.mean(X)) / np.std(X) # normalizacao
    
    
    for j in range(0, nrows):
	
	    Yn = np.dot(w.transpose(), X[j])
	    e = Y[j] - Yn
	    	    
	    w = w + alpha * e * X[j]

    Ynew = np.dot(X, w.transpose())
    Error = Y - Ynew
    EQM[i] = np.average(np.multiply(Error, Error))
    
    
print w

plt.plot(Y)
plt.plot(Ynew, 'r')
plt.show()

plt.plot(EQM)
plt.show()
