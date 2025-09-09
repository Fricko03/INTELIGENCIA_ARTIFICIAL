import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Universos de variables
x_flujo = np.arange(0, 101, 1)
x_densidad = np.arange(0, 101, 1)
x_tiempo = np.arange(0, 241, 1)

# Funciones de membresía Flujo
flujo_lo = fuzz.trapmf(x_flujo, [0,0,10,40])
flujo_md = fuzz.trimf(x_flujo, [30,50,70])
flujo_hi = fuzz.trapmf(x_flujo, [60,80,100,100])

# Funciones de membresía Densidad
densidad_lo = fuzz.trapmf(x_densidad, [0,0,10,40])
densidad_md = fuzz.trimf(x_densidad, [30,50,70])
densidad_hi = fuzz.trapmf(x_densidad, [60,80,100,100])

# Funciones de membresía Tiempo Verde
verde_corto = fuzz.trimf(x_tiempo, [0,30,60])
verde_medio = fuzz.trimf(x_tiempo, [50,100,150])
verde_largo = fuzz.trimf(x_tiempo, [120,180,240])

# Función para calcular activaciones y defuzzificación
def calcular_tiempo(flujo_val, densidad_val):
    # Grados de pertenencia
    f_lo = fuzz.interp_membership(x_flujo, flujo_lo, flujo_val)
    f_md = fuzz.interp_membership(x_flujo, flujo_md, flujo_val)
    f_hi = fuzz.interp_membership(x_flujo, flujo_hi, flujo_val)

    d_lo = fuzz.interp_membership(x_densidad, densidad_lo, densidad_val)
    d_md = fuzz.interp_membership(x_densidad, densidad_md, densidad_val)
    d_hi = fuzz.interp_membership(x_densidad, densidad_hi, densidad_val)

    # Reglas
    r1 = np.fmin(f_lo, d_lo)
    r2 = np.fmin(f_lo, d_md)
    r3 = np.fmin(f_lo, d_hi)
    r4 = np.fmin(f_md, d_lo)
    r5 = np.fmin(f_md, d_md)
    r6 = np.fmin(f_md, d_hi)
    r7 = np.fmin(f_hi, d_lo)
    r8 = np.fmin(f_hi, d_md)
    r9 = np.fmin(f_hi, d_hi)

    # Activaciones individuales
    act1 = np.fmin(r1, verde_corto)
    act2 = np.fmin(r2, verde_corto)
    act3 = np.fmin(r3, verde_medio)
    act4 = np.fmin(r4, verde_medio)
    act5 = np.fmin(r5, verde_medio)
    act6 = np.fmin(r6, verde_largo)
    act7 = np.fmin(r7, verde_medio)
    act8 = np.fmin(r8, verde_largo)
    act9 = np.fmin(r9, verde_largo)

    # Agregación final
    agregada = np.fmax.reduce([act1, act2, act3, act4, act5, act6, act7, act8, act9])

    # Defuzzificación
    verde_def = fuzz.defuzz(x_tiempo, agregada, 'centroid')
    rojo_def = 240 - verde_def
    tip_activation = fuzz.interp_membership(x_tiempo, agregada, verde_def)

    return verde_def, rojo_def, act1, act2, act3, act4, act5, act6, act7, act8, act9, agregada, tip_activation

# Ejemplo de prueba
flujo_actual = 40
densidad_actual = 90

verde, rojo, *acts, agregada, tip_activation = calcular_tiempo(flujo_actual, densidad_actual)

# Graficar
temp0 = np.zeros_like(x_tiempo)
plt.figure(figsize=(10,4))

# Funciones de membresía de salida
plt.plot(x_tiempo, verde_corto, 'b--', linewidth=0.8, label='Verde corto')
plt.plot(x_tiempo, verde_medio, 'g--', linewidth=0.8, label='Verde medio')
plt.plot(x_tiempo, verde_largo, 'r--', linewidth=0.8, label='Verde largo')

# Activaciones individuales
colors = ['b','c','m','y','orange','purple','brown','pink','grey']
for i, act in enumerate(acts):
    plt.fill_between(x_tiempo, temp0, act, color=colors[i], alpha=0.4)

# Activación agregada
plt.fill_between(x_tiempo, temp0, agregada, color='gold', alpha=0.7, label='Agregación final')

# Línea del valor defuzzificado
plt.plot([verde, verde], [0, tip_activation], 'k', linewidth=2, label='Valor defuzzificado')

plt.title(f"Flujo {flujo_actual}%, Densidad {densidad_actual}% → Verde {verde:.1f}s, Rojo {rojo:.1f}s")
plt.xlabel("Tiempo de semáforo (s)")
plt.ylabel("Grado de activación")
plt.legend()
plt.tight_layout()
plt.show()
