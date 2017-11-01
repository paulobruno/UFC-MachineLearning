# 388149 PAULO BRUNO DE SOUSA SERAFIM, 2016

# versao que usa sklearn


import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix


#Importando dados
with open('ex4data1.data') as txt:
	A = [[float(x) for x in line.split(",")] for line in txt]
txt.close()


# configuracao das matrizes de treinamento e teste
X = np.array(A) # converte array simples para array numpy de A

ncols = len(X[0])

#X = (X - np.mean(X)) / np.std(X) # normalizacao


# divisao por classes
X = np.split(X, [50, 100, 150]) # divide a matriz X em tres, 50 classe 1, 50 classe 2 e 50 classe 3

Xclass1 = X[0]
Xclass2 = X[1]
Xclass3 = X[2]


# divisao das matrizes de treinamento e teste
Xclass1 = np.split(Xclass1, [30, 50])
Xclass2 = np.split(Xclass2, [30, 50])
Xclass3 = np.split(Xclass3, [30, 50])

Xtraining = np.concatenate((Xclass1[0], Xclass2[0], Xclass3[0]))
Xtest = np.concatenate((Xclass1[1], Xclass2[1], Xclass3[1]))

np.random.shuffle(Xtraining) # embaralha as linhas de treinamento
np.random.shuffle(Xtest) # embaralha as linhas de teste


# configuracao dos atributos e classes
Ytraining = Xtraining[:, ncols-1] # Y eh a ultima coluna da matriz
Xtraining = Xtraining[:,:-1] # X sao todas as colunas ate a penultima

Ytest = Xtest[:, ncols-1] # Y eh a ultima coluna da matriz
Xtest = Xtest[:,:-1] # X sao todas as colunas ate a penultima

clf = QuadraticDiscriminantAnalysis()
clf.fit(Xtraining, Ytraining)

Ynew = clf.predict(Xtest)


print confusion_matrix(Ytest, Ynew)
