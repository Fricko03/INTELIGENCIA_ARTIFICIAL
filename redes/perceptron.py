import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
class PerceptronSimple:
    
    def __init__(self, learning_rate, epochs):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs
    

    def reporte(self):
        return self.weights, self.bias
    
        # Función de activación: Heaviside
    def activation(self, z):
        return np.heaviside(z, 0)
    
    def fit(self, X, y):
        n_features = X.shape[1]
        
        # Inicialización de parámetros (w y b)
        self.weights = np.zeros((n_features))
        self.bias = 0 # umbral
        
        # Iterar n épocas
        for epoch in range(self.epochs):
            
            # De a un dato a la vez
            for i in range(len(X)):
                z = np.dot(X, self.weights) + self.bias # Producto escalar de entradas y pesos + b
                y_pred = self.activation(z)             # Función de activación no lineal (Heaviside)
                
                #Actualización de pesos y bias
                self.weights +=  self.learning_rate * (y[i] - y_pred[i]) * X[i]
                self.bias += self.learning_rate * (y[i] - y_pred[i])
                
        return self.weights, self.bias
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
iris = load_iris() 

X = iris.data[:, (0, 1)] # petal length, petal width
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# plot
fig, ax = plt.subplots()
ax.scatter(X[:,1], X[:,0],c=y)
plt.show()
perceptron = PerceptronSimple(0.001, 100)

perceptron.fit(X_train, y_train)

pred = perceptron.predict(X_test)

print(accuracy_score(pred, y_test))

print(perceptron.reporte())
report = classification_report(pred, y_test, digits=2)
print(report)
sk_perceptron = Perceptron()
sk_perceptron.fit(X_train, y_train)
sk_perceptron_pred = sk_perceptron.predict(X_test)

# Accuracy

accuracy_score(sk_perceptron_pred, y_test)

report = classification_report(sk_perceptron_pred, y_test, digits=2)
print(report)