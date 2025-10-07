
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
boston_df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
boston_df.head() #esto es un print 


# buscamos valores nulos preprocesamiento 
print(boston_df.isnull().sum()) # --> NO hay valores nulos
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.hist(boston_df['medv'], bins=30)
plt.xlabel("Precio de las casas en miles de dolares")
plt.show()

# Vamos a ver las correlaciones entre el precio y los features
correlation_matrix = boston_df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)

plt.figure(figsize=(20, 10))

features = ['lstat', 'rm', 'ptratio', 'tax']
target = boston_df['medv']

for i, col in enumerate(features):
    plt.subplot(2, 2 , i+1)
    x = boston_df[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title("Variacion en los precios de la casas")
    plt.xlabel(col)
    plt.ylabel('"Precio de las casas en miles de dolares')
    
def myModel(X, y, random_state=1, scale=True, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    scaler = StandardScaler().fit(X_train)

    if scale:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) # cuidado, al aplicar el standard scaler, los datos dejan de ser dataframes

    n_train_samples, n_train_dim = X_train.shape


    # Para regresion utilizamos una unica capa oculta
    model = Sequential()
    model.add(Dense(128, input_dim=n_train_dim, activation='relu'))
    model.add(Dense(1, activation='linear'))


    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.summary()

    return model, X_train, X_test, y_train, y_test

def trainMyModel(model, X_train, y_train):

    result = model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=1)

    print("history.keys = ",result.history.keys())

    loss = result.history['loss']
    val_loss = result.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(7,7))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.show()


    mae = result.history['mae']
    val_mae = result.history['val_mae']


    plt.figure(figsize=(7,7))
    plt.plot(epochs, mae, 'y', label='Training MAE')
    plt.plot(epochs, val_mae, 'r', label='Validation MAE')
    plt.title('Training and validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()
    

def makePredictions(model, X_test, y_test):
    predictions = model.predict(X_test)

    y_test = np.array(y_test)

    predicted_value = np.array(predictions)
    predicted_value = predicted_value.reshape(max(predicted_value.shape), )

    return predicted_value

def plotPredictionsOnTwoAxes(predicted_value, X_test, y_test):

    data = np.array(X_test)  # ploteo con los datos sin estandarizar

    if (X_test.shape[1]==2) :
        data1 = data[:, 0]
        data2 = data[:, 1]
    else:
        data1 = data[:, 5]
        data2 = data[:, 12]
    true_value = np.array(y_test)
    plt.subplot(2, 1 , 1)
    plt.scatter(data1, true_value, marker='o', c='r')
    plt.scatter(data1, predicted_value, marker='o', c='b')
    plt.ylabel('"Precio de las casas en miles de dolares')
    plt.subplot(2, 1 , 2)
    plt.scatter(data2, true_value, marker='o', c='r')
    plt.scatter(data2, predicted_value, marker='o', c='b')
    plt.ylabel('"Precio de las casas en miles de dolares')
    plt.suptitle("Estimación de precios de casas - azul true value - red estimated value [DATOS DE TESTEO]")
    return data1, data2

def plotPredictionsSeparated(predicted_value, X_test, y_test):

    data = np.array(X_test)  # ploteo con los datos sin estandarizar

    if (X_test.shape[1]==2) :
        data1 = data[:, 0]
        data2 = data[:, 1]
    else:
        data1 = data[:, 5]
        data2 = data[:, 12]
    true_value = np.array(y_test)
    plt.subplot(2, 2 , 1)
    plt.scatter(data1, true_value, marker='o', c='r')
    plt.ylabel('Precio en miles de dolares')
    plt.title("True")
    plt.subplot(2, 2 , 2)
    plt.scatter(data1, predicted_value, marker='o', c='b')
    plt.ylabel('Precio en miles de dolares')
    plt.title("Predicted")
    plt.subplot(2, 2 , 3)
    plt.scatter(data2, true_value, marker='o', c='r')
    plt.ylabel('Precio en miles de dolares')
    plt.title("True")
    plt.subplot(2, 2 , 4)
    plt.scatter(data2, predicted_value, marker='o', c='b')
    plt.ylabel('Precio en miles de dolares')
    plt.title("Predicted")
    plt.suptitle("Estimación de precios de casas - azul true value - red estimated value [DATOS DE TESTEO]")
    return data1, data2

def plotTwoModels(data11, data12, true_value, predicted_value1, data21, data22, predicted_value2):
    plt.figure(figsize=(15,12))
    plt.subplot(2, 2 , 1)
    plt.scatter(data11, true_value, marker='o', c='r')
    plt.scatter(data11, predicted_value1, marker='o', c='b')
    plt.ylabel('Precio de las casas en miles de dolares')
    plt.title("True value - 2 parametros de entrenamiento - PRICE VS RM")
    plt.subplot(2, 2 , 2)
    plt.scatter(data12, true_value, marker='o', c='r')
    plt.scatter(data12, predicted_value1, marker='o', c='b')
    plt.ylabel('Precio de las casas en miles de dolares')
    plt.title("Predicted value - 2 parametros de entrenamiento - PRICE VS LSTAT")
    plt.suptitle("Estimación de precios de casas - azul true value - red estimated value [DATOS DE TESTEO]")
    plt.subplot(2, 2 , 3)
    plt.scatter(data21, true_value, marker='o', c='m')
    plt.scatter(data21, predicted_value2, marker='o', c='c')
    plt.ylabel('Precio de las casas en miles de dolares')
    plt.title("True value - 13 parametros de entrenamiento - PRICE VS RM")
    plt.subplot(2, 2 , 4)
    plt.scatter(data22, true_value, marker='o', c='m')
    plt.scatter(data22, predicted_value2, marker='o', c='c')
    plt.ylabel('"Precio de las casas en miles de dolares')
    plt.title("Predicted value - 13 parametros de entrenamiento - PRICE VS LSTAT")
    plt.suptitle("Estimación de precios de casas - Red entrenada con 2 parametros - Red Entrenada con 13 parametros")
    
X = boston_df[["rm", "lstat"]]
y = boston_df["medv"]  # no es lo mismo que tenga 1 par de corchetes o dos pares de corchetes
model, X_train, X_test, y_train, y_test = myModel(X, y, random_state=1, scale=False)
trainMyModel(model, X_train, y_train)
predicted_value = makePredictions(model, X_test, y_test)
data11, data12 = plotPredictionsOnTwoAxes(predicted_value, X_test, y_test)
# Prueba 2 --> 13 datos como features
X1 = boston_df[["crim", "zn" , "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]]
y1 = boston_df["medv"]  # no es lo mismo que tenga 1 par de corchetes o dos pares de corchetes
model1, X1_train, X1_test, y1_train, y1_test = myModel(X1, y1, random_state=1, scale=True)
trainMyModel(model1, X1_train, y1_train)
predicted_value1 = makePredictions(model1, X1_test, y1_test)
data21, data22 = plotPredictionsOnTwoAxes(predicted_value1, X1_test, y1_test)

plotPredictionsSeparated(predicted_value1, X1_test, y1_test)
plotTwoModels(data11, data12, y_test, predicted_value, data21, data22, predicted_value1)