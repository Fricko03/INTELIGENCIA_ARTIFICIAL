from matplotlib.pylab import rand
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from sklearn.cluster import KMeans
import numpy as np




def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

class fisRule:
    
            
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma

class fisInput:
    def __init__(self, min,max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids


    def view(self):
        x = np.linspace(self.minValue,self.maxValue,20)
        plt.figure()
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
            plt.plot(x,y)

class fis:
    def printRulesReadable(self):
        print("Reglas FIS encontradas (formato legible):")
        n_clusters = len(self.rules)
        n_inputs = len(self.inputs)
        
        for i, centroid in enumerate(self.rules):
            coef_start = i * (n_inputs + 1)
            coef_end = coef_start + n_inputs + 1
            rule_coefs = self.solutions[coef_start:coef_end]
            
            # Construir la ecuación lineal del consecuente
            terms = [f"{rule_coefs[j]:.4f}*x{j+1}" for j in range(n_inputs)]
            terms.append(f"{rule_coefs[-1]:.4f}")  # término independiente
            eq = " + ".join(terms)
            
            # Mostrar la regla
            conditions = " y ".join([f"x{j+1} ≈ {centroid[j]:.4f}" for j in range(n_inputs)])
            print(f"Regla {i+1}: Si {conditions} entonces y = {eq}")
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []



    def genfis(self, data, kmeans):

        start_time = time.time()
        r = kmeans.labels_
        cluster_center = kmeans.cluster_centers_

        print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)

        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        # print("nivel acti")
        # print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        # print("sumMu")
        # print(sumMu)
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]


        A = acti*inp/sumMu

        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        # print(solutions)
        return 0

    def evalfis(self, data):
        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)


    def viewInputs(self):
        for input in self.inputs:
            input.view()
#test genfis 1D
def my_exponential(x,A, B, C):
    return A*np.exp(-B*x)+C


#generacion con un 
x_normal=np.arange(-5,10,0.2)
y_normal=my_exponential(x_normal,2.0,0.5,5.0)

data = np.vstack((x_normal,y_normal)).T  # Nx2

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data)
r = kmeans.labels_
c = kmeans.cluster_centers_



y_ruidosa=y_normal+np.random.randn(len(x_normal))


m=np.vstack((x_normal, y_ruidosa)).T

data = np.vstack((x_normal,y_ruidosa)).T  # Nx2

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data)

r = kmeans.labels_
c = kmeans.cluster_centers_

#grafico de puntos con tres reglas/ cluster
plt.figure()
plt.subplot(1,2,1)
plt.plot(m[:,0],m[:,1])
plt.scatter(c[:,0],c[:,1], c="k",marker='X')
plt.plot(x_normal, y_normal,"k",linestyle='--')


##resolucion con datos con ruido
fis2 = fis()
fis2.genfis(data,kmeans)
fis2.viewInputs()
r = fis2.evalfis(np.vstack(x_normal))
plt.subplot(1,2,2)
plt.plot(x_normal,y)
plt.plot(x_normal,r,linestyle='--')
plt.title("Modelado con datos ruidosos")
# print(fis2.solutions)

#####Evaluacion en un valor
x_nuevo = np.array([[0.480]])   # ojo: debe ser 2D (matriz 1x1)
# Evaluar el sistema
y_pred = fis2.evalfis(x_nuevo)
print("Entrada:", x_nuevo[0][0])
print("Salida predicha:", y_pred[0])




y_pred = fis2.evalfis(np.vstack(x_normal))
mse = mean_squared_error(y_ruidosa, y_pred)
print(f"Error cuadrático medio: {mse:.12f}")
plt.show()