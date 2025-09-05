import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
def gbellmf(x,a,b,c):
    return 1/(1+ np.abs((x-c)/a)**(2*b))

def mu_pequeno(x):
    return gbellmf(x,6,4,-10)
def mu_mediano(x):
    return gbellmf(x,4,4,0)
def mu_grande(x):
    return gbellmf(x,6,4,10)

#DEFINO LAS ECUACIONES DE SUGENO
def y1(x):
    return x**2+0.1*x+6.4
def y2(x):
    return x**2-0.5*x+4
def y3(x):
    return 1.8*x**2+x-2

x=np.linspace(-10,10,50)

mu1=mu_pequeno(x)
mu2=mu_mediano(x)
mu3=mu_grande(x)

#promedio ponderado
num = mu1*y1(x)+mu2*y2(x)+mu3*y3(x)
den= mu1+mu2+mu3
y_salida=num/den

# Gráficos
# ----------------------------
plt.figure(figsize=(12,5))

# Funciones de membresía
plt.subplot(1,2,1)
plt.plot(x, mu1, label="Pequeño")
plt.plot(x, mu2, label="Mediano")
plt.plot(x, mu3, label="Grande")
plt.title("Funciones de membresía de entrada")
plt.legend()

# Entrada/salida
plt.subplot(1,2,2)

plt.plot(x, y_salida, 'k', label="Salida difusa")
plt.title("Curva Entrada/Salida")
plt.legend()

def sugeno_output(x):
    mu1 = mu_pequeno(x)
    mu2 = mu_mediano(x)
    mu3 = mu_grande(x)

    num = mu1*y1(x) + mu2*y2(x) + mu3*y3(x)
    den = mu1 + mu2 + mu3

    return num/den if den != 0 else None
print(sugeno_output(0))
plt.show()