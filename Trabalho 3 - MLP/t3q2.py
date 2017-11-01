# 388149 PAULO BRUNO DE SOUSA SERAFIM, 2016


import numpy as np
import math
import matplotlib.pyplot as plt


numberOfHiddenNeurons = 20
learningRate = 0.05
numberOfTrainings = 50


# defining phi function and vectorizing to use across all the array
def phi(x):
    return 1 / (1 + math.exp(-0.01 * x))

phi = np.vectorize(phi)

def derivativePhi(x):
    return phi(x) * (1.0 - phi(x))

derivativePhi = np.vectorize(derivativePhi)


#Importando dados
with open('newdata2.txt') as txt:
	A = [[float(x) for x in line.split(",")] for line in txt]
txt.close()


# configuracao das matrizes de treinamento e teste

X = np.array(A) # converte array simples para array numpy de A

nrows = len(X)
ncols = len(X[0])

np.random.shuffle(X) # embaralha as linhas

Y = X[:, ncols-1] # Y eh a ultima coluna da matriz
X = X[:,:-1] # X sao todas as colunas ate a penultima


X = (X - np.mean(X)) / np.std(X) # normalizacao

minusOnesCol = np.ones((nrows,1)) # cria coluna de um's
minusOnesCol = np.multiply(-1.0, minusOnesCol) # transforma em coluna de menos um's
X = np.append(minusOnesCol, X, axis=1) # adiciona coluna de menos um's


# divisao das matrizes de treinamento e teste

X = np.split(X, [306, 406, 506]) # divide a matriz X em tres, 4000 dados treinamento, 500 validacao e 500 teste
Xtraining = X[0] # primeira parte eh de treinamento
Xvalidation = X[1] # segunda eh de validacao
Xtest = X[2] # ultima eh de testes

Y = np.split(Y, [306, 406, 506]) # analogo
Ytraining = Y[0]
Yvalidation = Y[1]
Ytest = Y[2]


# incializa pesos aleatorios entre 0 e 0.1
numberOfExamplesTraining = len(Xtraining)
numberOfExamplesValidation = len(Xvalidation)
numberOfExamplesTest = len(Xtest)
numberOfInputs = len(Xtraining[0])
numberOfOutputs = 1

w = np.random.uniform(0, 0.5, size=(numberOfInputs, numberOfHiddenNeurons)) # camada input -> oculta
m = np.random.uniform(0, 0.5, size=(numberOfHiddenNeurons+1, numberOfOutputs)) # camada oculta -> saida
#z = np.zeros(shape=(numberOfOutputs, numberOfHiddenNeurons)) # array auxiliar

yNewTraining = np.zeros(shape=(len(Ytraining), 1))
yNewTest = np.zeros(shape=(len(Ytest), 1))
yNewValidation = np.zeros(shape=(len(Yvalidation), 1))


# 
meanErrorSquareTraining = np.zeros(shape=(numberOfTrainings, 1))
meanErrorSquareValidation = np.zeros(shape=(numberOfTrainings, 1))
meanErrorSquareTest = np.zeros(shape=(numberOfTrainings, 1))

errorByExampleTraining = np.zeros(shape=(numberOfExamplesTraining, 1))
errorByExampleValidation = np.zeros(shape=(numberOfExamplesValidation, 1))
errorByExampleTest = np.zeros(shape=(numberOfExamplesTest, 1))

uTraining = np.zeros(shape=(numberOfExamplesTraining, numberOfHiddenNeurons))
uValidation = np.zeros(shape=(numberOfExamplesValidation, numberOfHiddenNeurons))
uTest = np.zeros(shape=(numberOfExamplesTest, numberOfHiddenNeurons))

# loop de treinamento
for j in range (0, numberOfTrainings):

    for i in range (0, numberOfExamplesTraining):

        # da entrada pra camada oculta
        uTraining[i] = np.dot(Xtraining[i], w)
        z = phi(uTraining[i])
        z = np.insert(z, 0, -1)

        # da camada oculta pra saida
        y = np.dot(z, m)
        yNewTraining[i] = y

        # erro
        errorByExampleTraining[i] = Ytraining[i] - y
        
        # backpropagation
        sigmaOutput = errorByExampleTraining[i] * derivativePhi(y)
        so = np.reshape(sigmaOutput, (1, 1))
        zt = np.reshape(z, (len(z), 1))
        lalala = np.dot(zt, so)
        m = m + learningRate * lalala # atualilzacao dos pesos oculta -> saida

        sigmaInput = derivativePhi(uTraining[i]) * np.sum(sigmaOutput * m)
        si = np.reshape(sigmaInput, (1, len(sigmaInput)))
        Xt = np.reshape(Xtraining[i], (len(Xtraining[0]),1))
        w = w + learningRate * np.dot(Xt, si) # atualilzacao dos pesos entrada -> oculta

    errorByExampleTraining = errorByExampleTraining * errorByExampleTraining
    meanErrorSquareTraining[j] = np.average(errorByExampleTraining)
    
    #print (meanErrorSquareTraining[j])

    # validacao
    for i in range (0, numberOfExamplesValidation):
    
        # da entrada pra camada oculta
        uValidation[i] = np.dot(Xvalidation[i], w)
        z = phi(uValidation[i])
        z = np.insert(z, 0, -1)
    
        # da camada oculta pra saida
        y = np.dot(z, m)
        yNewValidation[i] = y
     
        # erro
        errorByExampleValidation[i] = Yvalidation[i] - y
        
    errorByExampleValidation = errorByExampleValidation * errorByExampleValidation
    meanErrorSquareValidation[j] = np.average(errorByExampleValidation)

    #print (meanErrorSquareValidation[j])

    print (str(j * 100 / numberOfTrainings) + "%")
    
plt.plot(meanErrorSquareValidation)
plt.plot(meanErrorSquareTraining)
plt.show()


# teste
for i in range (0, numberOfExamplesTest):

    # da entrada pra camada oculta
    uTest[i] = np.dot(Xtest[i], w)
    z = phi(uTest[i])
    z = np.insert(z, 0, -1)

    # da camada oculta pra saida
    y = np.dot(z, m)
    yNewTest[i] = y
	
 
plt.plot(Ytest, 'red')
plt.plot(yNewTest)
plt.show()

#    errorByExampleTest[i] = Ytest[i] - yNewTest[i]	
#
#errorByExampleTest = errorByExampleTest * errorByExampleTest
#meanErrorSquareTest[j] = np.average(errorByExampleTest)
