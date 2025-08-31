from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
# print(iris.DESCR)
# @title

# creo centros 
center_1 = np.array([1,1])
center_2 = np.array([5,5])
center_3 = np.array([8,1])

# Genero sets de datos al rededor de cada centro que cree
data_1 = np.random.randn(200, 2) + center_1
data_2 = np.random.randn(200,2) + center_2
data_3 = np.random.randn(200,2) + center_3

#concateno varios arreglos en uno 
#axis = indica que lo une por fila uno abajo del otro
data = np.concatenate((data_1, data_2, data_3), axis = 0) 
#scatter grafico
# data[:,0] : todo y 0 es porque es la primer columna

# plt.scatter(data[:,0], data[:,1], s=15)
# #figure es para crear otro grafico
# plt.show()

# @title
# Number of clusters
k = 3
# Number of training data
n = data.shape[0] #devuelve el tamaño
# Number of features in the data
c = data.shape[1] # dimension D

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
centers = np.random.randn(k,c)*std + mean

# Plot the data and the centers generated as random
# plt.scatter(data[:,0], data[:,1], s=7)
# plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
# plt.show()
# aca empieza el algoritmo
# @title
centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers


clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        distances[:,i] = np.linalg.norm(data - centers_new[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
#tengo un vector de tamaño n el cual hace referencia a cada dato 
# y ahi para cada dato i busco el en que cluster tiene una menor 
# distancia haciendo argmin axis=1 ya que necesito que me traiga el indice j
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)

plt.figure()
plt.scatter(data[:,0],data[:,1],s=7,c="r")
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)
plt.show()