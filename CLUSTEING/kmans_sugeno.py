import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Datos
datos = np.array([5,6,4,7,9,6,7,4,8,16,9,14,9,15,7,11,22,17,23,27,
       28,30,31,37,32,42,35,47,44,29,52,70,150,98,109,118,112,139,
       98,105,142,143,13,254,97,108,97,89,86,73,72,66,65,57,45,46,
       43,38,26,25,25,15,15,9,0,11,3,0,1,0,0,0,1,0,0,0,1,0,0,0,0,
       0,0,0,0,0,0,0])

X = datos.reshape(-1, 1)

def fuzzy_cmeans_plot_with_radii_per_cluster(X, n_clusters):
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T,
        c=n_clusters,
        m=2,
        error=0.005,
        maxiter=1000,
        seed=42
    )
    
    # Sigma por cluster
    sigmas = []
    for i in range(n_clusters):
        distancias = u[i]**2 * (datos - cntr[i].item())**2
        sigma_cluster = np.sqrt(np.sum(distancias) / np.sum(u[i]**2))
        sigmas.append(sigma_cluster)
    
    # Graficar gaussianas
    x_range = np.linspace(min(datos), max(datos), 500)
    plt.figure(figsize=(8,5))
    for i in range(n_clusters):
        c_val = cntr[i].item()
        gauss = np.exp(-((x_range - c_val)**2) / (2 * sigmas[i]**2))
        plt.plot(x_range, gauss, label=f'Cluster {i+1} (c={c_val:.2f}, σ={sigmas[i]:.2f})')
    
    plt.scatter(datos, np.zeros_like(datos), color='k', alpha=0.3, label='Datos reales')
    plt.title(f"Fuzzy C-Means con {n_clusters} clusters")
    plt.xlabel("Casos COVID")
    plt.ylabel("Grado de pertenencia")
    plt.legend()
    plt.show()
    
    return cntr, sigmas

# Ejecutar
centros2, sigmas2 = fuzzy_cmeans_plot_with_radii_per_cluster(X, 2)
centros3, sigmas3 = fuzzy_cmeans_plot_with_radii_per_cluster(X, 3)

print("Centros (2 clusters):", centros2.flatten())
print("Sigmas (2 clusters):", sigmas2)
print("Centros (3 clusters):", centros3.flatten())
print("Sigmas (3 clusters):", sigmas3)

# def fuzzy_cmeans_plot_with_fixed_sigma(X, n_clusters, radii=1):
#     # Aplicar FCM
#     cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
#         X.T,
#         c=n_clusters,
#         m=2,
#         error=0.005,
#         maxiter=1000,
#         seed=42
#     )
    
#     # Sigma fijo por cluster según la fórmula
#     sigma = radii * (np.max(datos) - np.min(datos)) / np.sqrt(8)
#     sigmas = [sigma]*n_clusters  # mismo sigma para todos los clusters
    
#     # Graficar gaussianas
#     x_range = np.linspace(min(datos), max(datos), 500)
#     plt.figure(figsize=(8,5))
#     for i in range(n_clusters):
#         c_val = cntr[i].item()
#         gauss = np.exp(-((x_range - c_val)**2) / (2 * sigma**2))
#         plt.plot(x_range, gauss, label=f'Cluster {i+1} (c={c_val:.2f}, σ={sigma:.2f})')
    
#     plt.scatter(datos, np.zeros_like(datos), color='k', alpha=0.3, label='Datos reales')
#     plt.title(f"Fuzzy C-Means con {n_clusters} clusters")
#     plt.xlabel("Casos COVID")
#     plt.ylabel("Grado de pertenencia")
#     plt.legend()
#     plt.show()
    
#     return cntr, sigmas

# # Ejecutar
# centros2, sigmas2 = fuzzy_cmeans_plot_with_fixed_sigma(X, 2)
# centros3, sigmas3 = fuzzy_cmeans_plot_with_fixed_sigma(X, 3)

# print("Centros (2 clusters):", centros2.flatten())
# print("Sigmas (2 clusters):", sigmas2)
# print("Centros (3 clusters):", centros3.flatten())
# print("Sigmas (3 clusters):", sigmas3)
