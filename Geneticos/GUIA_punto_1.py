from operator import index
import random
import pygad
import numpy as np
def binario_a_x(individuo):
    # Convierte binario a entero
    valor = int("".join(map(str, individuo)), 2)
    # Mapear al intervalo {0,2,4,...,30}
    return valor * 2

# Parámetros
num_individuos = 6
num_bits = 5  # porque tenemos 16 posibles valores
random.seed(1)
# Generar población inicial
poblacion = []
for _ in range(num_individuos):
    individuo = [random.randint(0, 1) for _ in range(num_bits)]
    poblacion.append(individuo)
print(poblacion,sep="\n")

# 1.1. Crear aleatoriamente los individuos que compondrán la población inicial.
# 1.2. Calcular para cada uno el valor de la función de evaluación 𝑓𝑖(𝑥) y la probabilidad de selección 𝑓𝑖(𝑥)/ ∑𝑘 𝑓𝑘(𝑥).
# Reconocer el mejor individuo de esta población inicial.
# 1.3. Se configura que 2 individuos pasarán por elite a la generación siguiente. Además, los dos mejores individuos generarán 4
# hijos por cruzamiento, con puntos de cruce en los bits 2 y 3. Generar los hijos.
# 1.4. Suponer que según la probabilidad de mutación dada se cambia el bit 2 del tercer hijo. Implementarla.
# 1.5. Calcular nuevamente las funciones de evaluación. Elegir el mejor individuo de esta nueva generación. ¿Se obtuvo una
# mejora respecto al mejor individuo de la generación anterior? Calcular el valor promedio de la función de evaluación.

def fitness(pobla):
    resultados= np.array([(300-(binario_a_x(i)-15)**2) for i in pobla])
    probs=np.array([ i/sum(resultados) for i in resultados])
    return resultados,probs

def ga(individuos):
    
    mejores_individuos=individuos
    mejor_fitness_global = -np.inf
    mejor_individuo_global=None
    for i in range(4):
        resultados,probs=fitness(mejores_individuos)
        idx = np.argmax(resultados)
        mejor_fitness_gen=resultados[idx]
        mejor_individuo_gen = mejores_individuos[idx]
        
        if mejor_fitness_gen > mejor_fitness_global:
            mejor_fitness_global = mejor_fitness_gen
            mejor_individuo_global = mejor_individuo_gen
            
            
        print(f"\nGeneración {i}")
        print("Población y fitness originales:")
        for ind, fit in zip(mejores_individuos, resultados):
            print(f"{ind} -> x={binario_a_x(ind)} -> fitness={fit}")
            
        mejores_indices = np.argsort(resultados)[-2:]  # Índices de los dos mejores
        mejores_individuos = [mejores_individuos[i] for i in mejores_indices]  # Acceso directo a los individuos originales
        #crossover
        p1=mejores_individuos[0]
        p2=mejores_individuos[1] # [0,1][1,0][0]
        h1=p1[:2] + p2[2:3+1] + p1[3+1:]
        h2=p1[:2] + p2[2:3+1] + p2[3+1:]
        h3=p2[:2] + p1[2:3+1] + p1[3+1:]
        h4=p2[:2] + p1[2:3+1] + p2[3+1:]
        
        mejores_individuos=[p1,p2,h1,h2,h3,h4]
        print("Poblacion luego cru")
        resultados,probs=fitness(mejores_individuos)
        for ind, fit in zip(mejores_individuos, resultados):
            print(f"{ind} -> x={binario_a_x(ind)} -> fitness={fit}")
        fines_original=(300-(binario_a_x(mejores_individuos[4])-15)**2)
        print(h3,binario_a_x(h3),fines_original)
        h3_mutado=h3.copy()
        h3_mutado[2]=1-h3_mutado[2]
        fitness_mutacion=(300-(binario_a_x(h3)-15)**2)
        print(h3,binario_a_x(h3), fitness_mutacion)
        if(fitness_mutacion>fines_original):
            print("hubo mejora")
        else:
            print("no hubo mejora")
    print(f"Mejor individuo{mejor_individuo_global,binario_a_x(mejor_individuo_global)} Con este fitness {mejor_fitness_global}")
ga(individuos=poblacion)