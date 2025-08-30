# # import pandas as pd
# # pd.set_option("display.max_columns", None)
# # pd.set_option("display.max_rows", None)
# # url = "https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv"
# # df = pd.read_csv(url)
# # df.describe().to_csv("resumen.csv")
# # pd.set_option("display.width", 1000)   # Ancho máximo de la tabla

# # print(df.describe())
# # Vamos a crear dos array de 5 filas x 4 columnas
# import numpy as np
# lista_1=[ "MANZANA","PERA","UVA","KIWI"]
# X3= np.array(lista_1).reshape(2,-1)
# X1 = np.arange(20).reshape(5, 4)
# X2 = np.arange(10, 30).reshape(5,4)
# print("------------------------------------------------------------------------")
# print("X1 ")
# print(X1)
# print("------------------------------------------------------------------------")
# print("X2")
# print(X2)
# print("------------------------------------------------------------------------")
# print("X3")
# print(X3,X3.shape,X3.ndim)
# print("------------------------------------------------------------------------")
# X3.append(3)
# print(X3)
import numpy as np
import matplotlib.pyplot as plt


X = np.linspace(0,10,50)

Y = np.array([x**2 + 2*x for x in X])

plt.ion()  # activa modo interactivo

plt.figure()
plt.plot(X, Y)
plt.title("Grafico unido")

plt.figure()
plt.plot(X, Y, '.k')
plt.title("Grafico con puntitos")
plt.grid()

plt.ioff()  # opcional: desactiva modo interactivo
plt.show()
print("hola")
# plt.figure()
# plt.plot(X, Y)
# plt.title("Grafico unido ")
# plt.grid()

# plt.figure()
# plt.plot(X, Y, '.k')

# plt.title("Grafico con puntitos")
# plt.grid()
# plt.show()

# # plt.figure(figsize=(10,4))

# # plt.subplot(1,2,1)
# # plt.plot(X, Y)
# # plt.title("Gráfico unido")

# # plt.subplot(1,2,2)
# # plt.plot(X, Y, '*')
# # plt.title("Gráfico con puntitos")
# # plt.grid()

# # plt.show()