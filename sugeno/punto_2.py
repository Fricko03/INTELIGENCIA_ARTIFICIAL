from matplotlib import pyplot as plt
import numpy as np
def gbllmf(x,a,b,c):
    return 1/(1+abs((x-c)/a)**(2*b))

def peque(x):
    return gbllmf(x,6,4,-10)

def med(x):
    return gbllmf(x,4,4,0)

def grande(x):
    return gbllmf(x,6,4,10)

#reglas
def y1(x):
    return x**2+0.1*x+6.4

def y2(x):
    return x**2-0.50*x+4

def y3(x):
    return 1.8*x**2+x-2

x=np.arange(-10,11,0.1)


mu1=peque(x)
mu2=med(x)
mu3=grande(x)
numerador=mu1*y1(x)+mu2*y2(x)+mu3*y3(x)
denominador=mu1+mu2+mu3
y_prima=numerador/denominador


#grafico
plt.figure(figsize=(12,5))

#plotear fundiones de membresia
plt.subplot(1,2,1)
plt.plot(x, mu1, label="Pequeño")
plt.plot(x, mu2, label="Mediano")
plt.plot(x, mu3, label="Grande")
plt.title("Funciones de membresía de entrada")
def sugeno_output(x):
    mu1 = peque(x)
    mu2 = med(x)
    mu3 = grande(x)

    num = mu1*y1(x) + mu2*y2(x) + mu3*y3(x)
    den = mu1 + mu2 + mu3

    return num/den if den != 0 else None
x0=-9.72
y0=sugeno_output(x0)
plt.subplot(1,2,2)
plt.plot(x, y_prima, 'k', label="Salida difusa")
plt.scatter(x0, y0, color='red', zorder=5)  # Marca el punto
plt.axhline(y0, color='red', linestyle='--')  # Línea horizontal hasta el eje y
plt.axvline(x0, color='red', linestyle='--')  # Línea vertical hasta el eje x
plt.title("Curva Entrada/Salida")
plt.legend()


##evaluar

print(y0)
x=-9.56
while (abs(sugeno_output(x)-100)>0.1):
    print(f"{sugeno_output(x)}\t{x}")
    x-=0.001
    
    

print(x)
plt.show()