# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import MinMaxScaler
# import time

# import numpy as np
# import matplotlib.pyplot as plt

# from scipy.spatial import distance_matrix

# def subclust2(data, Ra=1.5, Rb=0, AcceptRatio=0.2, RejectRatio=0.09):
#     if Rb==0:
#         Rb = Ra*1.1

#     scaler = MinMaxScaler()
#     scaler.fit(data)
#     ndata = scaler.transform(data)

#     P = distance_matrix(ndata,ndata)
#     alpha=(Ra/2)**2
#     P = np.sum(np.exp(-P**2/alpha),axis=0)

#     centers = []
#     i=np.argmax(P)
#     C = ndata[i]
#     p=P[i]
#     centers = [C]

#     continuar=True
#     restarP = True
#     while continuar:
#         pAnt = p
#         if restarP:
#             P=P-p*np.array([np.exp(-np.linalg.norm(v-C)**2/(Rb/2)**2) for v in ndata])
#         restarP = True
#         i=np.argmax(P)
#         C = ndata[i]
#         p=P[i]
#         if p>AcceptRatio*pAnt:
#             centers = np.vstack((centers,C))
#         elif p<RejectRatio*pAnt:
#             continuar=False
#         else:
#             dr = np.min([np.linalg.norm(v-C) for v in centers])
#             if dr/Ra+p/pAnt>=1:
#                 centers = np.vstack((centers,C))
#             else:
#                 P[i]=0
#                 restarP = False
#         if not any(v>0 for v in P):
#             continuar = False
#     distancias = [[np.linalg.norm(p-c) for p in ndata] for c in centers]
#     labels = np.argmin(distancias, axis=0)
#     centers = scaler.inverse_transform(centers)
#     return labels, centers

# datos=pd.read_csv(r"C:\Users\tomas\Documents\Diodo.csv",sep=";",header=0)
# m=datos.iloc[:,0:].values

# data_x = datos["x"].values   # columna de entrada
# data_y = datos["y"].values   # columna de salida

# data   = np.vstack((data_x, data_y)).T

# r,c = subclust2(m)
# print(r,c)

# plt.figure()
# plt.scatter(m[:,0],m[:,1], c=r)
# plt.scatter(c[:,0],c[:,1], marker='X')
# plt.show()




# def gaussmf(data, mean, sigma):
#     return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

# class fisRule:
#     def __init__(self, centroid, sigma):
#         self.centroid = centroid
#         self.sigma = sigma

# class fisInput:
#     def __init__(self, min,max, centroids):
#         self.minValue = min
#         self.maxValue = max
#         self.centroids = centroids


#     def view(self):
#         x = np.linspace(self.minValue,self.maxValue,20)
#         plt.figure()
#         for m in self.centroids:
#             s = (self.minValue-self.maxValue)/8**0.5
#             y = gaussmf(x,m,s)
#             plt.plot(x,y)

# class fis:
#     def __init__(self):
#         self.rules=[]
#         self.memberfunc = []
#         self.inputs = []



#     def genfis(self, data, radii):

#         start_time = time.time()
#         labels, cluster_center = subclust2(data, radii)

#         print("--- %s seconds ---" % (time.time() - start_time))
#         n_clusters = len(cluster_center)

#         cluster_center = cluster_center[:,:-1]
#         P = data[:,:-1]
#         #T = data[:,-1]
#         maxValue = np.max(P, axis=0)
#         minValue = np.min(P, axis=0)

#         self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
#         self.rules = cluster_center
#         self.entrenar(data)

#     def entrenar(self, data):
#         P = data[:,:-1]
#         T = data[:,-1]
#         #___________________________________________
#         # MINIMOS CUADRADOS (lineal)
#         sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
#         f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

#         nivel_acti = np.array(f).T
#         print("nivel acti")
#         print(nivel_acti)
#         sumMu = np.vstack(np.sum(nivel_acti,axis=1))
#         print("sumMu")
#         print(sumMu)
#         P = np.c_[P, np.ones(len(P))]
#         n_vars = P.shape[1]

#         orden = np.tile(np.arange(0,n_vars), len(self.rules))
#         acti = np.tile(nivel_acti,[1,n_vars])
#         inp = P[:, orden]


#         A = acti*inp/sumMu

#         # A = np.zeros((N, 2*n_clusters))
#         # for jdx in range(n_clusters):
#         #     for kdx in range(nVar):
#         #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
#         #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

#         b = T

#         solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
#         self.solutions = solutions #.reshape(n_clusters,n_vars)
#         print(solutions)
#         return 0

#     def evalfis(self, data):
#         sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
#         f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
#         nivel_acti = np.array(f).T
#         sumMu = np.vstack(np.sum(nivel_acti,axis=1))

#         P = np.c_[data, np.ones(len(data))]

#         n_vars = P.shape[1]
#         n_clusters = len(self.rules)

#         orden = np.tile(np.arange(0,n_vars), n_clusters)
#         acti = np.tile(nivel_acti,[1,n_vars])
#         inp = P[:, orden]
#         coef = self.solutions

#         return np.sum(acti*inp*coef/sumMu,axis=1)


#     def viewInputs(self):
#         for input in self.inputs:
#             input.view()
# #test genfis 1D
# def my_exponential(A, B, C, x):
#     return A*np.exp(-B*x)+C



# plt.plot(data_x, data_y,linestyle='--')


# data = np.vstack((data_x, data_y)).T

# fis2 = fis()
# fis2.genfis(data, 1.1)

# fis2.viewInputs()
# r = fis2.evalfis(np.vstack(data_x))

# plt.figure()
# plt.plot(data_x,data_y)
# plt.plot(data_x,r,linestyle='--')

# print(fis2.solutions)

# # r1 = data_x*-2.29539539+ -41.21850973
# # r2 = data_x*-15.47376916 -79.82911266
# # r3 = data_x*-15.47376916 -79.82911266
# # plt.plot(data_x,r1)
# # plt.plot(data_x,r2)
# # plt.plot(data_x,r3)

# plt.show()

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
datos=pd.read_csv(r"C:\Users\tomas\Documents\Diodo.csv",sep=";",header=0)
m=datos.iloc[:,0:].values

data_x = datos["x"].values   # columna de entrada
data_y = datos["y"].values   # columna de salida

data = np.vstack((data_x, data_y)).T  # Nx2

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)
r = kmeans.labels_
c = kmeans.cluster_centers_

# print("Etiquetas:", labels)
# print("Centros:", centers)

# data   = np.vstack((data_x, data_y)).T

# r,c = subclust2(m)
# print(r,c)

plt.figure()
plt.scatter(m[:,0],m[:,1], c=r)
plt.scatter(c[:,0],c[:,1], marker='X')
plt.show()




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
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []



    def genfis(self, data, radii):

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
        print("nivel acti")
        print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        print("sumMu")
        print(sumMu)
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
        print(solutions)
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
def my_exponential(A, B, C, x):
    return A*np.exp(-B*x)+C



plt.plot(data_x, data_y,linestyle='--')


data = np.vstack((data_x, data_y)).T

fis2 = fis()
fis2.genfis(data, 1.1)

fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_x))

plt.figure()
plt.plot(data_x,data_y)
plt.plot(data_x,r,linestyle='--')

print(fis2.solutions)

# r1 = data_x*-2.29539539+ -41.21850973
# r2 = data_x*-15.47376916 -79.82911266
# r3 = data_x*-15.47376916 -79.82911266
# plt.plot(data_x,r1)
# plt.plot(data_x,r2)
# plt.plot(data_x,r3)


x_nuevo = np.array([[0.480]])   # ojo: debe ser 2D (matriz 1x1)

# Evaluar el sistema
y_pred = fis2.evalfis(x_nuevo)

print("Entrada:", x_nuevo[0][0])
print("Salida predicha:", y_pred[0])
plt.show()