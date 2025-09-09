
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
# Generamos los universos de variables
#   * Calidad de comida y servicio son rangos subjetivos [0, 10]
#   * La propina tiene un rango de 0 a 25 en puntos percentuales
x_flujo = np.arange(0, 101, 0.1)       # 0 a 100%
x_densidad = np.arange(0, 101, 0.1)    # 0 a 100%
x_tiempo = np.arange(0, 241, 1)      # 0 a 240 segundos

# Funciones de membresía flujo (%)
flujo_lo = fuzz.trapmf(x_flujo, [0, 0, 10, 25])
flujo_md = fuzz.trimf(x_flujo, [15, 30, 45])
flujo_hi = fuzz.trapmf(x_flujo, [35, 50, 100, 100])

# Funciones de membresía densidad (%)
densidad_lo = fuzz.trapmf(x_densidad, [0, 0, 10, 30])
densidad_md = fuzz.trimf(x_densidad, [20, 50, 80])
densidad_hi = fuzz.trapmf(x_densidad, [70, 90, 100, 100])

# Funciones de membresía tiempo (segundos)
tiempo_corto = fuzz.trimf(x_tiempo, [0, 30, 60])
tiempo_medio = fuzz.trimf(x_tiempo, [50, 120, 180])
tiempo_largo = fuzz.trimf(x_tiempo, [150, 210, 240])

# Graficamos la calidad de la comida
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_flujo, flujo_lo, 'b', linewidth=1.5, label='Baja')
ax0.plot(x_flujo, flujo_md, 'g', linewidth=1.5, label='Media')
ax0.plot(x_flujo, flujo_hi, 'r', linewidth=1.5, label='Alta')
ax0.set_title('Densidad de autos')
ax0.legend()

plt.tight_layout()
# Graficamos la calidad del servicio
fig, ax1 = plt.subplots(figsize=(8, 3))

ax1.plot(x_densidad, densidad_lo, 'b', linewidth=1.5, label='Baja')
ax1.plot(x_densidad, densidad_md, 'g', linewidth=1.5, label='Media')
ax1.plot(x_densidad, densidad_hi, 'r', linewidth=1.5, label='Alta')
ax1.set_title('Densidad de personas')
ax1.legend()

plt.tight_layout()

# Graficamos el porcentaje de propina
fig, ax2 = plt.subplots(figsize=(8, 3))

ax2.plot(x_tiempo, tiempo_corto, 'b', linewidth=1.5, label='Poco')
ax2.plot(x_tiempo, tiempo_medio, 'g', linewidth=1.5, label='Medio')
ax2.plot(x_tiempo, tiempo_largo, 'r', linewidth=1.5, label='Mucho')
ax2.set_title('Tiempo de semaforo en rojo')
ax2.legend()

plt.tight_layout()
# Necesitamos la activación de nuestras funciones de pertenencia difusa en estos valores.
# Los valores específicos 6.5 y 9.8 se toman de la entrada del usuario.
# Si tuvieramos la calidad de la comida y el servicio en variables, podríamos usarlos aquí.

# Ejemplo de valores puntuales
flujo_actual = 40    # porcentaje
densidad_actual = 90 # porcentaje

# Grados de pertenencia flujo
flujo_level_lo = fuzz.interp_membership(x_flujo, flujo_lo, flujo_actual)
flujo_level_md = fuzz.interp_membership(x_flujo, flujo_md, flujo_actual)
flujo_level_hi = fuzz.interp_membership(x_flujo, flujo_hi, flujo_actual)

# Grados de pertenencia densidad
densidad_level_lo = fuzz.interp_membership(x_densidad, densidad_lo, densidad_actual)
densidad_level_md = fuzz.interp_membership(x_densidad, densidad_md, densidad_actual)
densidad_level_hi = fuzz.interp_membership(x_densidad, densidad_hi, densidad_actual)


# Ahora tomamos nuestras reglas y las aplicamos. La regla 1 se refiere a la mala comida y al mal servicio.
active_rule1 = np.fmin(flujo_level_lo, densidad_level_lo)

# Ahora aplicamos esta regla a nuestra función de salida
# Si es mala comida o servicio, la propina será baja

tip_activation_lo = np.fmin(active_rule1, tiempo_corto)

# Para la regla 2, conectamos el servicio aceptable con una propina media
activation_rule_2=np.fmax(flujo_level_lo,densidad_level_hi)
tip_activation_md = np.fmin(activation_rule_2, tiempo_largo)

# Para la regla 3, conectamos la comida buena y servicio excelente con una propina alta
active_rule3 = np.fmin(flujo_level_md, densidad_level_md)
tip_activation_hi = np.fmin(active_rule3, tiempo_medio)
temp0 = np.zeros_like(x_tiempo)

# Graficamos las funciones de pertenencia de salida
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_tiempo, temp0, tip_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_tiempo, tiempo_corto, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_tiempo, temp0, tip_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_tiempo, tiempo_medio, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_tiempo, temp0, tip_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_tiempo, tiempo_largo, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Salida de las funciones de pertenencia difusa')

# Sacamos el eje superior y derecho
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
# Concatenamos las tres funciones de pertenencia de salida juntas
aggregated = np.fmax(tip_activation_lo,
                     np.fmax(tip_activation_md, tip_activation_hi))

# Calculamos el resultado difuso
tip = fuzz.defuzz(x_tiempo, aggregated, 'centroid')
tip_activation = fuzz.interp_membership(x_tiempo, aggregated, tip)

# Graficamos el resultado
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_tiempo, tiempo_corto, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_tiempo, tiempo_medio, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_tiempo, tiempo_largo, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_tiempo, temp0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Concatenación de las funciones de pertenencia difusa y resultado (línea)')

# Sacamos el eje superior y derecho
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

print(f"{tip:.2f}")

plt.show()


# import numpy as np
# import skfuzzy as fuzz
# import matplotlib.pyplot as plt

# # Universos de variables
# x_flujo = np.arange(0, 101, 1)        # Flujo de autos (%)
# x_densidad = np.arange(0, 101, 1)     # Densidad peatones (%)
# x_tiempo = np.arange(0, 241, 1)       # Tiempo de semáforo en segundos

# # Funciones de membresía Flujo
# flujo_lo = fuzz.trapmf(x_flujo, [0,0,10,40])
# flujo_md = fuzz.trimf(x_flujo, [30,50,70])
# flujo_hi = fuzz.trapmf(x_flujo, [60,80,100,100])

# # Funciones de membresía Densidad
# densidad_lo = fuzz.trapmf(x_densidad, [0,0,10,40])
# densidad_md = fuzz.trimf(x_densidad, [30,50,70])
# densidad_hi = fuzz.trapmf(x_densidad, [60,80,100,100])

# # Funciones de membresía Tiempo Verde
# verde_corto = fuzz.trimf(x_tiempo, [0,30,60])
# verde_medio = fuzz.trimf(x_tiempo, [50,100,150])
# verde_largo = fuzz.trimf(x_tiempo, [120,180,240])

# # Funciones de membresía Tiempo Rojo
# rojo_corto = fuzz.trimf(x_tiempo, [0,30,60])
# rojo_medio = fuzz.trimf(x_tiempo, [50,100,150])
# rojo_largo = fuzz.trimf(x_tiempo, [120,180,240])

# # Función para aplicar reglas
# def aplicar_reglas(flujo_val, densidad_val):
#     # Grados de pertenencia de entradas
#     f_lo = fuzz.interp_membership(x_flujo, flujo_lo, flujo_val)
#     f_md = fuzz.interp_membership(x_flujo, flujo_md, flujo_val)
#     f_hi = fuzz.interp_membership(x_flujo, flujo_hi, flujo_val)

#     d_lo = fuzz.interp_membership(x_densidad, densidad_lo, densidad_val)
#     d_md = fuzz.interp_membership(x_densidad, densidad_md, densidad_val)
#     d_hi = fuzz.interp_membership(x_densidad, densidad_hi, densidad_val)

#     # Reglas de tiempo verde (autos)
#     rule_v1 = np.fmin(f_lo, d_lo)    # Flujo bajo y pocos peatones → verde corto
#     rule_v2 = np.fmin(f_lo, d_md)    # Flujo bajo, densidad media → verde corto
#     rule_v3 = np.fmin(f_lo, d_hi)    # Flujo bajo, densidad alta → verde medio

#     rule_v4 = np.fmin(f_md, d_lo)    # Flujo medio, pocos peatones → verde medio
#     rule_v5 = np.fmin(f_md, d_md)    # Flujo medio, densidad media → verde medio
#     rule_v6 = np.fmin(f_md, d_hi)    # Flujo medio, densidad alta → verde largo

#     rule_v7 = np.fmin(f_hi, d_lo)    # Flujo alto, pocos peatones → verde medio
#     rule_v8 = np.fmin(f_hi, d_md)    # Flujo alto, densidad media → verde largo
#     rule_v9 = np.fmin(f_hi, d_hi)    # Flujo alto, densidad alta → verde largo

#     # Activaciones de salida verde
#     verde_activacion = np.fmax.reduce([
#         np.fmin(rule_v1, verde_corto),
#         np.fmin(rule_v2, verde_corto),
#         np.fmin(rule_v3, verde_medio),
#         np.fmin(rule_v4, verde_medio),
#         np.fmin(rule_v5, verde_medio),
#         np.fmin(rule_v6, verde_largo),
#         np.fmin(rule_v7, verde_medio),
#         np.fmin(rule_v8, verde_largo),
#         np.fmin(rule_v9, verde_largo)
#     ])

#     # Para tiempo rojo, podemos usar regla inversa
#     rojo_activacion = 240 - verde_activacion  # ciclo total 240 s

#     # Defuzzificación
#     verde_def = fuzz.defuzz(x_tiempo, verde_activacion, 'centroid')
#     rojo_def = 240 - verde_def

#     return verde_def, rojo_def, verde_activacion

# # --- Prueba de ejemplos ---
# pruebas = [
#     (20, 10),  # flujo bajo, densidad baja
#     (20, 80),  # flujo bajo, densidad alta
#     (60, 50),  # flujo medio, densidad media
#     (90, 90),  # flujo alto, densidad alta
# ]

# for flujo_val, densidad_val in pruebas:
#     verde, rojo, activacion = aplicar_reglas(flujo_val, densidad_val)
#     print(f"Flujo={flujo_val}%, Densidad={densidad_val}% → Verde={verde:.1f}s, Rojo={rojo:.1f}s")

#     # Graficar activación verde
#     plt.figure(figsize=(8,3))
#     plt.plot(x_tiempo, verde_corto, 'b--', label='Verde corto')
#     plt.plot(x_tiempo, verde_medio, 'g--', label='Verde medio')
#     plt.plot(x_tiempo, verde_largo, 'r--', label='Verde largo')
#     plt.fill_between(x_tiempo, 0, activacion, facecolor='orange', alpha=0.7)
#     plt.title(f"Flujo {flujo_val}%, Densidad {densidad_val}% → Verde {verde:.1f}s")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
