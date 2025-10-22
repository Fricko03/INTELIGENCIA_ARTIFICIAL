import skfuzzy as fuzz
import matplotlib.pyplot as plt 
import numpy as np


#universos de entrada
notas_examen = np.linspace(0,100,500)
conceptos = np.linspace(0,10,500)

#universo de salida
notas_finales = np.linspace(0,10,500)

#funciones de pertenencia para la nota de examen (entrada)
NE_pert_baja = fuzz.sigmf(notas_examen, 40, -0.3) # [0;50]
NE_pert_media = fuzz.gaussmf(notas_examen, 60, 10) # [45;75]
NE_pert_alta = fuzz.sigmf(notas_examen, 70, 0.3) # [70;100]

#funciones de pertenencia para el concepto (entrada)
# el concepto es fuzzy, le damos un rango de valores de 0 a 10
C_pert_regular = fuzz.sigmf(conceptos, 5.5, -2.5) # [0;6]
C_pert_bueno = fuzz.gaussmf(conceptos, 6.5, 1.2) # [5;8]
C_pert_excelente = fuzz.sigmf(conceptos, 7.5, 2.5) # [7;10]

#funciones de pertenencia para la nota final (salida)
NF_pert_baja = fuzz.sigmf(notas_finales, 4.2, -3) **2
NF_pert_bajaMedia = fuzz.gaussmf(notas_finales, 4.875, 1.1)
NF_pert_media = fuzz.gaussmf(notas_finales, 6, 1) **2
NF_pert_mediaAlta = fuzz.gaussmf(notas_finales, 7.75, 0.9)
NF_pert_alta = fuzz.sigmf(notas_finales, 8, 3)**2

#grafico de la nota de examen
fig, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(notas_examen, NE_pert_baja, 'b', linewidth=1.5, label='Baja')
ax1.plot(notas_examen, NE_pert_media, 'g', linewidth=1.5, label='Media')
ax1.plot(notas_examen, NE_pert_alta, 'r', linewidth=1.5, label='Alta')
ax1.set_title("Notas de exámen")
ax1.legend()
plt.tight_layout()
plt.show(block=False)

#grafico del concepto
fig, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(conceptos, C_pert_regular, 'b', linewidth=1.5, label='Regular')
ax1.plot(conceptos, C_pert_bueno, 'r', linewidth=1.5, label='Bueno')
ax1.plot(conceptos, C_pert_excelente, 'c', linewidth=1.5, label='Excelente')
ax1.set_title("Conceptos")
ax1.legend()
plt.tight_layout()
plt.show(block=False)

#grafico de las notas finales
fig, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(notas_finales, NF_pert_baja, 'b', linewidth=1.5, label='Baja')
ax1.plot(notas_finales, NF_pert_bajaMedia, 'g', linewidth=1.5, label='Baja-Media')
ax1.plot(notas_finales, NF_pert_media, 'y', linewidth=1.5, label='Media')
ax1.plot(notas_finales, NF_pert_mediaAlta, "pink", linewidth=1.5, label='Media-alta')
ax1.plot(notas_finales, NF_pert_alta, 'r', linewidth=1.5, label='Alta')
ax1.set_title("Notas finales")
ax1.legend()
plt.tight_layout()
plt.show()

#valores a analizar 
valores = [(80,10),(50,7),(100,8),(20,6),(25,7)]

for nota_examen, concepto in valores:
    #valores de verdad 
    NE_in_baja = fuzz.interp_membership(notas_examen, NE_pert_baja, nota_examen)
    NE_in_media = fuzz.interp_membership(notas_examen, NE_pert_media, nota_examen)
    NE_in_alta = fuzz.interp_membership(notas_examen, NE_pert_alta, nota_examen)
    
    C_in_regular = fuzz.interp_membership(conceptos, C_pert_regular, concepto)
    C_in_bueno = fuzz.interp_membership(conceptos, C_pert_bueno, concepto)
    C_in_excelente = fuzz.interp_membership(conceptos, C_pert_excelente, concepto)

    #activacion de reglas en los antecedenetes
    activacion_regla1 = np.fmin(min(NE_in_baja,C_in_regular),NF_pert_baja)
    activacion_regla2 = np.fmin(min(NE_in_baja,C_in_bueno),NF_pert_baja)
    activacion_regla3 = np.fmin(min(NE_in_baja,C_in_excelente),NF_pert_bajaMedia)
    activacion_regla4 = np.fmin(min(NE_in_media,C_in_regular),NF_pert_bajaMedia)
    activacion_regla5 = np.fmin(min(NE_in_media,C_in_bueno),NF_pert_media)
    activacion_regla6 = np.fmin(min(NE_in_media,C_in_excelente),NF_pert_mediaAlta)
    activacion_regla7 = np.fmin(min(NE_in_alta,C_in_regular),NF_pert_mediaAlta)
    activacion_regla8 = np.fmin(min(NE_in_alta,C_in_bueno),NF_pert_alta)
    activacion_regla9 = np.fmin(min(NE_in_alta,C_in_excelente),NF_pert_alta)

    #agregacion
    agregacion = np.fmax.reduce([activacion_regla1,activacion_regla2,activacion_regla3,activacion_regla4,activacion_regla5,activacion_regla6,activacion_regla7,activacion_regla8,activacion_regla9])

    #defuzzificacion
    notas_final = fuzz.defuzz(notas_finales, agregacion, 'centroid')
    print(f"Con {nota_examen} pts en el examen y {concepto:.2f} de concepto, el alumno obtiene una nota final de {notas_final:.2f} pts.")
