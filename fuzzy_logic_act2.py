from simpful import *
import matplotlib.pyplot as plt
import numpy as np

def gbellmf(x, a, b, c):
    return 1 / (1 + abs((x-c)/a)**(2*b))

FS = FuzzySystem()

# 1. Definir variable de entrada x con universo de discurso [-10, 10]
x = LinguisticVariable([
    FuzzySet(function=lambda x: gbellmf(x, 6, 4, -10), term="pequeno"),
    FuzzySet(function=lambda x: gbellmf(x, 4, 4, 0), term="mediano"),
    FuzzySet(function=lambda x: gbellmf(x, 6, 4, 10), term="grande")
], universe_of_discourse=[-10, 10])

FS.add_linguistic_variable("x", x)

# 2. Definir funciones de salida
FS.set_output_function("function1", "x**2 + 0.1*x + 6.4", ["x"])
FS.set_output_function("function2", "x**2 - 0.5*x + 4", ["x"])
FS.set_output_function("function3", "1.8*x**2 + x - 2", ["x"])

# 3. Definir reglas Sugeno de segundo orden
RULE_1 = "IF (x IS pequeno) THEN (z IS function1)"
RULE_2 = "IF (x IS mediano) THEN (z IS function2)"
RULE_3 = "IF (x IS grande) THEN (z IS function3)"
FS.add_rules([RULE_1, RULE_2, RULE_3])

# 4. Evaluar el sistema
x_vals = np.linspace(-10, 10, 200)
results = []

for val in x_vals:
    FS.set_variable('x', val)
    output = FS.Sugeno_inference()
    results.append(output['z'])

# 5. Graficar
plt.plot(x_vals, results, label="Salida Sugeno (2º orden)")
plt.xlabel("x")
plt.ylabel("z")
plt.title("FIS Sugeno con funciones de salida polinomiales")
plt.legend()
plt.grid()
plt.show()
