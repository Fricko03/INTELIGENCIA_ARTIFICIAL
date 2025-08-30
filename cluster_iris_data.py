from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()

# @title

# creo centros 
center_1 = np.array([1,1])
center_2 = np.array([5,5])
center_3 = np.array([8,1])


# Number of clusters
k = 3
# Number of training data
n = iris.data.shape[0]
# Number of features in the data
c = iris.data.shape[1]
print(c)
# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(iris.data, axis = 0)
std = np.std(iris.data, axis = 0)
centers = np.random.randn(k,c)*std + mean

# Plot the data and the centers generated as random
colors=['orange', 'blue', 'green']
for i in range(n):
    plt.scatter(iris.data[i, 0], iris.data[i,1], s=7, color = colors[iris.target[i]])
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

iris.data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

cont = 0

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        distances[:,i] = np.linalg.norm(iris.data - centers[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)

    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(iris.data[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
#esto esta para que corte
    cont += 1
    if cont == 100:
        print('break')
        break
centers_new
print("ciclos: ",cont)
plt.figure()
colors=['orange', 'blue', 'green']
for i in range(n):
    plt.scatter(iris.data[i, 0], iris.data[i,1], s=7, color = colors[iris.target[i]])
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)
plt.show()