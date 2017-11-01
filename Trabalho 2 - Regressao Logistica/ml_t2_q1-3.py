# PAULO BRUNO DE SOUSA SERAFIM      354086


import numpy as np
import math
import matplotlib.pyplot as plt


#Importando dados
with open('ex2data1.txt') as txt:
	A = [[float(x) for x in line.split(",")] for line in txt]
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


# k-fold
numberOfGroups = 10


# divisao das matrizes de treinamento e teste
numOfTrainingData = nrows / numberOfGroups

#X = np.split(X, numberOfGroups) # divide a matriz X nos subgrupos
#Y = np.split(Y, numberOfGroups) # analogo

X = X.tolist()
Y = Y.tolist()

wForEachTurn = np.zeros((numberOfGroups, ncols-1))
results = np.zeros(numberOfGroups)


for k in range(numberOfGroups):

    # pega um para teste e os outros para treinamento
    Xtraining = X[:k * numOfTrainingData] + X[(k+1) * numOfTrainingData:]
    Xtest = X[k * numOfTrainingData:][:numOfTrainingData]
    
    Ytraining = Y[:k * numOfTrainingData] + Y[(k+1) * numOfTrainingData:]
    Ytest = Y[k * numOfTrainingData:][:numOfTrainingData]
    
    
    # transformando em np.array    
    Xtraining = np.array(Xtraining)
    Xtest = np.array(Xtest)
    
    Ytraining = np.array(Ytraining)
    Ytest = np.array(Ytest)
    
    
    # treinamento

    nrows = len(Xtraining)
    ncols = len(Xtraining[0])

    a = np.zeros(ncols) # cria vetor linha de um's
    aT = a.transpose()

    alpha = 0.01
    ages = 1000

    EQM = np.zeros(ages)
    Ynew = np.zeros(nrows)


    for i in range(0, ages):
	    for j in range(0, nrows):
		
		    Yn = np.dot(Xtraining[j], aT)
		    h = 1 / (1 + math.exp(-Yn))
		
		    aT = aT + alpha * (Ytraining[j] - h) * Xtraining[j] 

            Ynew[j] = h
            	
	    Error = Ytraining - Ynew
	    EQM[i] = np.average(np.multiply(Error, Error))
        
	
    wForEachTurn[k] = aT
    print aT


    # execucao dos testes

    count = 0
    nrows = len(Xtest)

    for i in range (0, nrows):
	    result = np.dot(Xtest[i], aT)

	    if (result > 0.5):
		    YtestNew = 1
	    else:
		    YtestNew = 0
		
	    if(Ytest[i] == YtestNew):
		    count += 1

    results[k] = float(count) / float(nrows)
    #print "Percentage of success: " + str("%.2f" % ((float(count) / float(nrows)) * 100)) + "%"
    

wMean = np.mean(wForEachTurn, axis=0)

print results
print "Average percentage of success: " + str("%.2f" % (np.mean(results) * 100)) + "%"
print "Average weights: " + str(wMean)
