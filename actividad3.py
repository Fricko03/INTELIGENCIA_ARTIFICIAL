import numpy as np
from matplotlib import pyplot as plt

def y(x, A, B, C):
    return A * np.exp(-1*B*x) + C

#reglas PREGUNTAR COMO DEFINIRLAS
#si la curva baja rapido -> a mas negativo
#si la curva se aplana -> a mas cercano a 0
#b aprox el valor medio del intervalo de pertenencia
def r1(x): #[-1; 3]
    return  -0.7*x + 6.7

def r2(x): #[2; 7]
    return  -0.3*x + 6.

def r3(x): #[6; 10]
    return  -.1*x + 5.2

#funciones de membresia
def mu_baja(x):    
    return np.exp(-((x - 2)**2) / (2 * 1.5**2)) #lo centro en 2

def mu_media(x):   
    return np.exp(-((x - 5)**2) / (2 * 1.5**2)) #lo centro en 5

def mu_alta(x):    
    return np.exp(-((x - 8)**2) / (2 * 1.5**2)) #lo centro en 8

def ecm(y1, y2):
   return np.mean((y2 - y1)**2)

#-------------------------------
if __name__ == "__main__":
    #parametros
    A = 2.0
    B = 0.5
    C = 5.0
    #conjunto de datos
    x = np.linspace(-1,10,50)
    #salida
    y_salida_real = y(x,A,B,C)
    
    #3.1
    #grafico
    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.plot(x, y_salida_real, color='green', label="x")
    plt.title("Curva Entrada/Salida real")
    #plt.show()

    #3.2
    mu1 = mu_baja(x)
    mu2 = mu_media(x)
    mu3 = mu_alta(x)
    num = mu1*r1(x) + mu2*r2(x) + mu3*r3(x)
    den = mu1 + mu2 + mu3
    y_salida_estimada = num/den #normalizo

    errorEst = ecm(y_salida_real, y_salida_estimada)
    print(f"ECM con estimacion: {errorEst:.4f}")
    errorNorm = errorEst / np.var(y_salida_real)
    #no sabia si ese error era bueno o no asique chat me dijo que haga el error normalizado:
    print(f"ECM con estimacion NORMALIZADO (<0.1 excelente | <0.5 bueno | >1 mal): {errorNorm:.4f}")
    
    plt.subplot(1,3,2)
    plt.plot(x, mu1, label="Baja")
    plt.plot(x, mu2, label="Media")
    plt.plot(x, mu3, label="Alta")
    plt.title("Funciones de membresía de entrada")
    plt.legend()

    # Entrada/salida
    plt.subplot(1,3,3)
    plt.plot(x, y_salida_estimada, color='red', label="Salida difusa")
    plt.title("Curva Entrada/Salida estimada")
    plt.legend()

    plt.show()

    #3.3
    np.random.seed(42)  
    ruido = np.random.randn(len(x))  
    y_ruidoso = y_salida_real + ruido
    errorRuido = ecm(y_salida_real, y_ruidoso)
    print(f"ECM con ruido: {errorRuido:.4f}")
    print

    #curva real con ruido
    plt.subplot(1,2,1)
    plt.plot(x, y_salida_real, label="Real sin ruido")
    plt.plot(x, y_ruidoso, label="Real con ruido")
    plt.title("Curva Real con Ruido")
    plt.legend()

    #estimacion vs ruidosa
    #PREGUNTAR QUE DEBO ANALIZAR EN EL GRAFICO DEL RUIDO????
    plt.subplot(1,2,2)
    plt.plot(x, y_ruidoso, label="Real con ruido")
    plt.plot(x, y_salida_estimada, label="Estimacion difusa", color='red')
    plt.title("Modelo Sugeno vs Datos Ruidosos")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #3.4
    x_interpolados = np.linspace(-1,10,200) #pongo mas puntos
    y_interpolados = y(x_interpolados,A,B,C)

    mu1 = mu_baja(x_interpolados)
    mu2 = mu_media(x_interpolados)
    mu3 = mu_alta(x_interpolados)
    num = mu1*r1(x_interpolados) + mu2*r2(x_interpolados) + mu3*r3(x_interpolados)
    den = mu1 + mu2 + mu3
    y_salida_estimada_interpolados = num/den

    plt.figure(figsize=(8, 5))
    #plt.plot(x_interpolados, y_interpolados, label='Datos interpolados', color='blue')    
    plt.plot(x_interpolados, y_salida_estimada_interpolados, label='Datos interpolados estimados', color='red')
    plt.plot(x_interpolados, y_interpolados, label='Datos originales', color='green')
    plt.title('Interpolacion del modelo con nuevos valores de x')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



"""
Extraer conclusiones: ¿es conveniente que el modelo aproxime los datos ruidosos exactamente? ¿siempre un modelo cuyo 
error de entrenamiento es casi nulo es mejor que otro con mayor error? ¿qué ocurre al aumentar el número de reglas del 
modelo de Sugeno?
"""



