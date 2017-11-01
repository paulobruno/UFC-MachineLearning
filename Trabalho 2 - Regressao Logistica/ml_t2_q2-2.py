# PAULO BRUNO DE SOUSA SERAFIM      354086


import numpy as np
import math
import matplotlib.pyplot as plt


#Importando dados
with open('newdata2.txt') as txt:
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


# divisao das matrizes de treinamento e teste
numOfTrainingData = 70

X = np.split(X, [numOfTrainingData,]) # divide a matriz X em duas
Xtraining = X[0] # primeira metade eh de treinamento
Xtest = X[1] # segunda eh de testes

Y = np.split(Y, [numOfTrainingData,]) # analogo
Ytraining = Y[0]
Ytest = Y[1]


# treinamento

nrows = len(Xtraining)
ncols = len(Xtraining[0])

a = np.zeros(ncols) # cria vetor linha de um's
aT = a.transpose()

alpha = 0.01
ages = 1000

EQM = np.zeros(ages)
Ynew = np.zeros(nrows)

l = 0 # valor do lamba
lambdaI = l * np.identity(len(aT))
lambdaI[0][0] = 0


for i in range(0, ages):    
	for j in range(0, nrows):
		
		Yn = np.dot(Xtraining[j], aT)
		h = 1 / (1 + math.exp(-Yn))
		
		aT = aT + alpha * ( ((Ytraining[j] - h) * Xtraining[j]) - np.dot(lambdaI, aT) )

        Ynew[j] = h
        	
	Error = Ytraining - Ynew
	EQM[i] = np.average(np.multiply(Error, Error))
    
	
print aT

plt.plot(EQM)
plt.show()


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


print "Percentage of success: " + str("%.2f" % ((float(count) / float(nrows)) * 100)) + "%"


# saida em arquivo

output = open("theta.txt", "w")
np.savetxt(output, aT, fmt="%.8f")
output.close()
