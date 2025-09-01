from statistics import kde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
iris = pd.read_csv("https://raw.githubusercontent.com/M-Yerro/IA2024/main/02b-IA2024%20irisdata.csv", header=0, names=["sepal_length","sepal_width",
                                                                                                                       "petal_length",
                                                                                                                "petal_width",
                                                                                                                "species"])
x = iris.iloc[:, [0, 1, 2, 3]].values
y=iris.loc[:,["sepal_length","petal_width"]].values
# print(iris.info)

#Frequency distribution of species"
iris_outcome = pd.crosstab(
       index=iris["species"],  # Make a crosstab
                              columns="count")    
# Name the count column
#se crea una tabla con columna contador y como fina cada uno de los tipos 
#clustering clasificado???

iris_setosa=iris.loc[iris["species"]=="Iris-setosa"]
#clasifica los distintos tipos
iris_virginica=iris.loc[iris["species"]=="Iris-virginica"]
iris_versicolor=iris.loc[iris["species"]=="Iris-versicolor"]
# sns.FacetGrid(iris,hue="species").map(
#        sns.histplot,
#        "petal_length",
#        kde=True,
#        edgecolor=None,
#        stat="density",
       
#        kde_kws=dict(cut=2)
#        ).add_legend()

# sns.FacetGrid(iris,hue="species").map(
#        sns.histplot,
#        "sepal_length",
#        kde=True,
#        edgecolor=None,
#        stat="density",
       
#        kde_kws=dict(cut=2)
#        ).add_legend()
# sns.FacetGrid(iris,hue="species").map(
#        sns.histplot,
#        "sepal_width",
#        kde=True,
#        edgecolor=None,
#        stat="density",
       
#        kde_kws=dict(cut=2)
#        ).add_legend()
# sns.FacetGrid(iris,hue="species").map(
#        sns.histplot,
#        "petal_width",
#        kde=True,
#        edgecolor=None,
#        stat="density",
       
#        kde_kws=dict(cut=2)
#        ).add_legend()
# print(iris)
# sns.scatterplot(data=iris, x="sepal_length", y="sepal_width",
#                 hue="species", palette=['orange','blue','green'], s=7)


# plt.show()

#Finding the optimum number of clusters for k-means classification

wcss = []
# busca el numero de cluster que mejor se adapta al problema 
# metodo del codo 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)

    wcss.append(kmeans.inertia_)
    
# plt.plot(range(1, 11), wcss)
# plt.title('The elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS') #within cluster sum of squares


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans=kmeans.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.show()
# 3d scatterplot using matplotlib

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.show()