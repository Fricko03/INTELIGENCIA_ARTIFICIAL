
from sklearn.cluster import KMeans
import numpy as np
data = np.loadtxt(r"C:\Users\tomas\INTELIGENCIA_ARTIFICIAL\FUZZY\diodo.txt")  # columna 0: V, columna 1: I

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)  # data es Nx2, input y output
centros = kmeans.cluster_centers_
labels = kmeans.labels_

print("Centros:", centros)
from sklearn.linear_model import LinearRegression

# Para cada cluster, ajustamos una función lineal
reglas = []
for i in range(2):
    cluster_data = data[labels == i]
    X = cluster_data[:,0].reshape(-1,1)
    y = cluster_data[:,1]
    
    modelo = LinearRegression().fit(X, y)
    reglas.append((X.mean(), modelo.coef_[0], modelo.intercept_))  # centro y coef lineal

print("Reglas:", reglas)

import skfuzzy as fuzz
from skfuzzy import control as ctrl

x = ctrl.Antecedent(np.linspace(min(data[:,0]), max(data[:,0]), 100), 'x')
y = ctrl.Consequent(np.linspace(min(data[:,1]), max(data[:,1]), 100), 'y', defuzzify_method='centroid')

# Definir conjuntos difusos para cada centro
for i, (centro, a, b) in enumerate(reglas):
    x[str(i)] = fuzz.gaussmf(x.universe, centro, 1)  # sigma = 1 como ejemplo
    y[str(i)] = lambda x_val: a*x_val + b
    