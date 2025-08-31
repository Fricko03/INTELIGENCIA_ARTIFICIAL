# https://github.com/M-Yerro/IA2025/blob/main/02a-IA2025%20datosReduccion.txt

from curses.panel import bottom_panel
from re import X
from statistics import kde
from turtle import left
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import KMeans


def sample_representative(X, n_clusters=3, fraction=0.2, random_state=42):
    """
    Toma un subconjunto representativo de los datos X basado en clustering.
    
    Parámetros:
        X : np.array, shape (n_samples, n_features)
        n_clusters : int, cantidad de clusters a usar
        fraction : float, porcentaje de puntos a tomar por cluster
        random_state : int, reproducibilidad

    Retorna:
        X_subset : np.array con los puntos más representativos
    """
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    representative_idx = []
    representative_labels=[]
    representative_intraclusterdistance=[]
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        n_take = max(1, int(len(cluster_points) * fraction))  # al menos 1
        closest_idx = np.argsort(distances)[:n_take]          # más cercanos al centroide
        original_idx = np.where(labels == i)[0][closest_idx]
        representative_idx.extend(original_idx)
        distances=np.linalg.norm(X[original_idx] - centroids[i], axis=1)**2
        representative_intraclusterdistance.append(sum(distances))
        representative_labels.extend([i] * len(original_idx))
        
    labels_subset = np.array(representative_labels)
    X_subset = X[representative_idx]
    
    
   

    return X_subset,labels_subset,sum(representative_intraclusterdistance)
iris = pd.read_csv("https://raw.githubusercontent.com/M-Yerro/IA2025/refs/heads/main/02a-IA2025%20datosReduccion.txt",
                   sep="\\s+",header=None, names=["X","Y"])

x=iris.iloc[:, :].values


plt.xlim(-1, iris["X"].max() + 1)
plt.ylim(-1, iris["Y"].max() + 1)
plt.xticks(np.arange(-1, iris["X"].max()+2, 1))
plt.yticks(np.arange(-1, iris["Y"].max()+2, 1))
plt.grid(True, linestyle='--', alpha=0.5)
clusterer_whith_all=KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
cluster_labels=clusterer_whith_all.fit_predict(x)

 # 2nd Plot showing the actual clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)

plt.scatter(x[:, 0], x[:, 1], marker=".", s=20, lw=0, alpha=0.7, c=colors, edgecolor="k")

# Labeling the clusters
clustering_whith_all = clusterer_whith_all.cluster_centers_
# Draw white circles at cluster centers
plt.scatter(
    clustering_whith_all[:, 0],
    clustering_whith_all[:, 1],
    marker="o",
    c="white",
    alpha=1,
    s=200,
    edgecolor="k",
)

for i, c in enumerate(clustering_whith_all):
    plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

plt.title("The visualization of the clustered data.")
plt.xlabel("X")
plt.ylabel("Y")

clustering_with_20_percent,clus_label,representative_intraclusterdistance = sample_representative(x, n_clusters=3, fraction=0.2)
print("Cantidad original:", len(x))
print("Cantidad representativa:", len(clustering_with_20_percent))
color_2 = cm.nipy_spectral((clus_label.astype(float)) / 3)
color_3=["red","blue","green"]
plt.figure()
plt.xlim(-1, iris["X"].max() + 1)
plt.ylim(-1, iris["Y"].max() + 1)
plt.xticks(np.arange(-1, iris["X"].max()+2, 1))
plt.yticks(np.arange(-1, iris["Y"].max()+2, 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.scatter(clustering_with_20_percent[:, 0], clustering_with_20_percent[:, 1], marker=".", s=20, c=[color_3[i] for i in clus_label],lw=0, alpha=0.7, edgecolor="k")
plt.scatter(
    clustering_whith_all[:, 0],
    clustering_whith_all[:, 1],
    marker="o",
    c="white",
    alpha=1,
    s=200,
    edgecolor="k",
)

for i, c in enumerate(clustering_whith_all):
    plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
plt.show()


# distancia intraclustaer
intra_distances=[]
for i in range(len(clustering_whith_all)):
    cluster_points=x[cluster_labels==i]
    distances = (np.linalg.norm(cluster_points - clustering_whith_all[i], axis=1)**2)
    intra_distances.append(sum(distances))
print("Inertia total:", clusterer_whith_all.inertia_)
print(f"Intracluster distance whit all {sum(intra_distances):.2f}")
print(f"Intracluster distance whit 20% {representative_intraclusterdistance:.2f}")