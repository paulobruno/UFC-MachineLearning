# PAULO BRUNO DE SOUSA SERAFIM      388149


import numpy as np
import matplotlib.pyplot as plt


# importacao de dados

with open('ex1data1.txt') as txt:
    A = [[float(x) for x in line.split(',')] for line in txt]
txt.close()


# configuracao das matrizes de treinamento e teste

M = np.array(A) # converte array simples para array numpy de A

nrows = len(M)
ncols = len(M[0]) - 1


# inicializacao treinamento
w0 = 1
w1 = 1

alpha = 0.001
age = 1000

EQM = np.zeros(age)


# treinamento
for i in range(0, age):
    
    np.random.shuffle(M) # embaralha as linhas

    X = M[:, 0]
    Y = M[:, ncols]
    Ynew = w1 * X + w0
    Error = Y - Ynew
    
    w0 = w0 + alpha * np.average(Error)
    w1 = w1 + alpha * np.average(Error * X)

    EQM[i] = np.average(np.multiply(Error, Error))
    

print w0
print w1

plt.plot(X, Y, 'bo')
plt.plot(X, Ynew, 'r')
plt.show()

plt.plot(EQM)
plt.show()
