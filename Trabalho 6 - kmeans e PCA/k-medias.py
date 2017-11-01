import numpy as np
from random import randint
import matplotlib.pyplot as plt
import math

#definicao da funcao de calculo de distancia entre um ponto e um centroide
def dist(ponto, centroide):
	return math.sqrt(math.pow(ponto[0] - centroide[0], 2) + math.pow(ponto[1] - centroide[1], 2) + math.pow(ponto[2] - centroide[2], 2) + math.pow(ponto[3] - centroide[3], 2))

#Funcao que escolhe K centroides na amostra
def definirCentroides(k, pontos):
    centroides = np.zeros((k, 4)) 
    #Escolhendo centrois aleatorios
    for i in range(0, k):
	    centroides[i] = pontos[randint(0, len(pontos) - 1)]
    return centroides
    
#Funcao que define em qual grupo esta cada ponto
def definirGrupos(pontos, grupos, centroides):     
    candidato = 9999
    for i in range(0, len(pontos)):
	    for j in range(0, len(centroides)):
		     novaDistancia = dist(pontos[i], centroides[j])
		     if(novaDistancia < candidato):
		     	candidato = novaDistancia
		     	grupos[i] = j
	    candidato = 9999
    return grupos
 
#Funcao que recalcula centroides de acordo com os membros de cada grupo
def recalculaCentroides(pontos, grupo, centroides):
    soma = np.array([0, 0, 0, 0])
    contador = 0
    for i in range(0, len(centroides)):
        for j in range(0, len(pontos)):
            if(grupo[j] == i):
                soma = soma + pontos[j] 
                contador = contador + 1  
        centroides[i] = soma / contador            
        contador = 0 
        soma = np.array([0, 0, 0, 0])
        
    return centroides

def convergir(centroides, centroidesAntigos, distanciaMinima):
    for i in range(0 , len(centroides)):
        if(dist(centroides[i], centroidesAntigos[i]) > distanciaMinima):
            return True
    return False	
#################  inicio do algoritmo  ################### 

#Lendo arquivo
with open('teste.data') as txt:
	pontos = [[float(x) for x in line.split(',')] for line in txt]
txt.close()


#Lendo arquivo
with open('teste2.data') as txt:
	yreal = [[float(x) for x in line.split(',')] for line in txt]
txt.close()

#numero de centroides 
k = 4
soma = np.zeros(k+1) # cria vetor de soma's

#for z in range(2, k):
for z in range(k, k+1):
    print ""
    print "Execucao com " + str(z) + " grupo(s)"
    #K centroides aleatorios da amostra  
    centroides = definirCentroides(z, pontos)

    #array de para salvar em qual grupo esta cada ponto da amostra 
    grupo = np.ones((len(pontos), 1))

    centroidesAntigos = centroides + 1

    while (convergir(centroides, centroidesAntigos, 0.05)):
        #Definindo em que grupo esta cada ponto
        grupo = definirGrupos(pontos, grupo, centroides)
        
        #guardando valor dos centroides antes do recalculo para verificacao de convergencia 
        np.copyto(centroidesAntigos, centroides)    
        
        #recalculando centroides de acordo com a media dos pontos de seu grupo
        centroides = recalculaCentroides(pontos, grupo, centroides)
        #print dist(centroides[1], centroidesAntigos[1])

    contador = 0    
    for i in range(0, len(centroides)):
        for j in range(0, len(pontos)):
            if(grupo[j] == i):
                contador = contador + 1
                soma[z] += dist(pontos[j], centroides[i])
            #print "elemento: " + str(pontos[j]) + "  classe: " + str(i)
        print str(contador) + " membros no grupo " + str(i)
        contador = 0
        
    print "Soma das distancias: " + str(soma[z])
            
print soma

#plt.plot(soma, '-bo')
#plt.xlim([2,5])
#plt.ylim([50, 140])
#plt.show()
