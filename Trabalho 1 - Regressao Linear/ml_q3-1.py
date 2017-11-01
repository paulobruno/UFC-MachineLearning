# PAULO BRUNO DE SOUSA SERAFIM      388149


import numpy as np
import matplotlib.pyplot as plt


# importacao de dados
with open('ex1data3.txt') as txt:
    A = [[float(x) for x in line.split()] for line in txt]
txt.close()


# configuracao das matrizes de treinamento e teste
M = np.array(A) # converte array simples para array numpy de A

nrows = len(M)

onesCol = np.ones((nrows,1)) # cria coluna de um's
M = np.append(onesCol, M, axis=1) # adiciona coluna de um's

ncols = len(M[0])


# atribuicao matrizes
#np.random.shuffle(M) # embaralha as linhas

Y = M[:, ncols-1] # Y eh a ultima coluna da matriz
X = M[:,:-1] # X sao todas as colunas ate a penultima

X = (X - np.mean(X)) / np.std(X) # normalizacao


# divisao das matrizes de treinamento e teste
numOfTrainingData = 30

X = np.split(X, [numOfTrainingData,]) # divide a matriz X em duas
Xtraining = X[0] # primeira metade eh de treinamento
Xtest = X[1] # segunda eh de testes

Y = np.split(Y, [numOfTrainingData,]) # analogo
Ytraining = Y[0]
Ytest = Y[1]


# treinamento
XtrainingT = Xtraining.transpose()

lambdas = 50

w = np.ones(ncols-1)
EQMtraining = np.zeros(lambdas)
EQMtest = np.zeros(lambdas)


for l in range(0, lambdas):

    lambdaI = l * np.identity(len(XtrainingT))
    lambdaI[0][0] = 0

    w = np.dot(XtrainingT, Xtraining) + lambdaI
    w = np.linalg.inv(w)
    w = np.dot(w, XtrainingT)
    w = np.dot(w, Ytraining)

    # print w
    

    # execucao teste
    Ynew = np.dot(Xtest, w)

    #plt.plot(Ytest)
    #plt.plot(Ynew, 'r')
    #plt.show()

    Error = Ytest - Ynew
    EQMtest[l] = np.average(np.multiply(Error, Error))


    # EQM treinamento
    Ynew = np.dot(Xtraining, w)
    Error = Ytraining - Ynew
    EQMtraining[l] = np.average(np.multiply(Error, Error))


plt.plot(EQMtest)
plt.plot(EQMtraining, 'r')
plt.show()
