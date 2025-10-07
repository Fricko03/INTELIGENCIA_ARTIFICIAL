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
        return np.heaviside(z, 1)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for epoch in range(self.epochs):
            # shuffle each epoch (good practice)
            indices = np.random.permutation(n_samples)
            X_shuf = X[indices]
            y_shuf = y[indices]
            for xi, yi in zip(X_shuf, y_shuf):
                z_i = np.dot(xi, self.weights) + self.bias
                y_pred_i = 1 if z_i >= 0 else 0
                update = self.learning_rate * (yi - y_pred_i)
                self.weights += update * xi
                self.bias += update
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
plt.show()