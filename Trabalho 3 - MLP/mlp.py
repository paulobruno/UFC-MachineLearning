# 388149 PAULO BRUNO DE SOUSA SERAFIM, 2016


import numpy as np
import math
import matplotlib.pyplot as plt


numberOfHiddenNeurons = 20
learningRate = 0.15
numberOfTrainings = 150


# defining phi function and vectorizing to use across all the array
def phi(x):
    return 1 / (1 + math.exp(-1.0 * x))

phi = np.vectorize(phi)

def derivativePhi(x):
    return phi(x) * (1.0 - phi(x))

derivativePhi = np.vectorize(derivativePhi)


#Importando dados
with open('test.txt') as txt:
	A = [[float(x) for x in line.split(",")] for line in txt]
txt.close()


# configuracao das matrizes de treinamento e teste

X = np.array(A) # converte array simples para array numpy de A

nrows = len(X)
ncols = len(X[0])

np.random.shuffle(X) # embaralha as linhas


X = np.hsplit(X, [400, 410])

Y = X[1]
X = X[0]

#TODO: verificar se a normalizacao deve vir antes ou depois de adicionar a coluna de -1
X = (X - np.mean(X)) / np.std(X) # normalizacao 

minusOnesCol = np.ones((nrows,1)) # cria coluna de um's
minusOnesCol = np.multiply(-1.0, minusOnesCol) # transforma em coluna de menos um's
X = np.append(minusOnesCol, X, axis=1) # adiciona coluna de menos um's


# divisao das matrizes de treinamento e teste

X = np.split(X, [4000, 4500, 5000]) # divide a matriz X em tres, 4000 dados treinamento, 500 validacao e 500 teste
Xtraining = X[0] # primeira parte eh de treinamento
Xvalidation = X[1] # segunda eh de validacao
Xtest = X[2] # ultima eh de testes

Y = np.split(Y, [4000, 4500, 5000]) # analogo
Ytraining = Y[0]
Yvalidation = Y[1]
Ytest = Y[2]


# incializa pesos aleatorios entre 0 e 0.1
numberOfExamplesTraining = len(Xtraining)
numberOfExamplesValidation = len(Xvalidation)
numberOfExamplesTest = len(Xtest)
numberOfInputs = len(Xtraining[0])
numberOfOutputs = len(Ytraining[0])

w = np.random.uniform(0, 0.1, size=(numberOfInputs, numberOfHiddenNeurons)) # camada input -> oculta
m = np.random.uniform(0, 0.1, size=(numberOfHiddenNeurons+1, numberOfOutputs)) # camada oculta -> saida
z = np.empty(shape=(numberOfOutputs, numberOfHiddenNeurons)) # array auxiliar

yNewTraining = np.empty(shape=(len(Ytraining), len(Ytraining[0])))
yNewTest = np.empty(shape=(len(Ytest), len(Ytest[0])))
yNewValidation = np.empty(shape=(len(Yvalidation), len(Yvalidation[0])))


# 
meanErrorSquareTraining = np.empty(shape=(numberOfTrainings, 1))
meanErrorSquareValidation = np.empty(shape=(numberOfTrainings, 1))
meanErrorSquareTest = np.empty(shape=(numberOfTrainings, 1))

errorByExampleTraining = np.empty(shape=(numberOfExamplesTraining, len(Ytraining[0])))
errorByExampleValidation = np.empty(shape=(numberOfExamplesValidation, len(Ytest[0])))
errorByExampleTest = np.empty(shape=(numberOfExamplesTest, len(Yvalidation[0])))


# loop de treinamento
for j in range (0, numberOfTrainings):

    for i in range (0, numberOfExamplesTraining):

        # da entrada pra camada oculta
        u = np.dot(Xtraining[i], w)
        z = phi(u)
        z = np.insert(z, 0, -1)

        # da camada oculta pra saida
        y = np.dot(z, m)
        yNewTraining[i] = phi(y)

        # erro
        errorByExampleTraining[i] = Ytraining[i] - yNewTraining[i]
        
        # backpropagation
        sigmaOutput = errorByExampleTraining[i] * derivativePhi(y)
        so = np.reshape(sigmaOutput, (1, len(sigmaOutput)))
        zt = np.reshape(z, (len(z), 1))
        m = m + learningRate * np.dot(zt, so) # atualilzacao dos pesos oculta -> saida

        sigmaInput = derivativePhi(u) * np.sum(sigmaOutput * m)
        si = np.reshape(sigmaInput, (1, len(sigmaInput)))
        Xt = np.reshape(Xtraining[i], (len(Xtraining[0]),1))
        w = w + learningRate * np.dot(Xt, si) # atualilzacao dos pesos entrada -> oculta

    errorByExampleTraining = errorByExampleTraining * errorByExampleTraining
    meanErrorSquareTraining[j] = np.average(errorByExampleTraining)
    
    #print (meanErrorSquareTraining[j])

    # validacao
    for i in range (0, numberOfExamplesValidation):
    
        # da entrada pra camada oculta
        u = np.dot(Xvalidation[i], w)
        z = phi(u)
        z = np.insert(z, 0, -1)
    
        # da camada oculta pra saida
        y = np.dot(z, m)
        yNewValidation[i] = phi(y)
    	
        # erro
        errorByExampleValidation[i] = Yvalidation[i] - yNewValidation[i]
        
    errorByExampleValidation = errorByExampleValidation * errorByExampleValidation
    meanErrorSquareValidation[j] = np.average(errorByExampleValidation)

    #print (meanErrorSquareValidation[j])

    print (str(j * 100 / numberOfTrainings) + "%")
    
plt.plot(meanErrorSquareValidation)
plt.plot(meanErrorSquareTraining)
plt.show()

plt.plot(meanErrorSquareValidation, 'o')
plt.plot(meanErrorSquareTraining, 'o')
plt.show()

# teste
okSum = 0

for i in range (0, numberOfExamplesTest):

    # da entrada pra camada oculta
    u = np.dot(Xtest[i], w)
    z = phi(u)
    z = np.insert(z, 0, -1)

    # da camada oculta pra saida
    y = np.dot(z, m)
    yNewTest[i] = phi(y)
	
    
    for k in range (0, len(Ytest[0])):
        if (Ytest[i][k] == 1):
            if (yNewTest[i][k] > 0.5):
                okSum += 1

print (str(okSum * 100 / numberOfExamplesTest) + "%")
#    errorByExampleTest[i] = Ytest[i] - yNewTest[i]	
#
#errorByExampleTest = errorByExampleTest * errorByExampleTest
#meanErrorSquareTest[j] = np.average(errorByExampleTest)



# saida em arquivo
#output = open("output.txt", "w")
#np.savetxt(output, zip(Ytest, yNewTest), fmt="%.4f")
#output.close()
