import numpy as np
from tensorflow import keras
##keras interprete 
from keras.models import Sequential #(siempre alimentar al de adelante )
from keras.layers import Dense
from keras.layers import Flatten #de 2 dimensiones a 1 
from keras.layers import Dropout #(apagar neuronas)
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

plt.rcParams["figure.figsize"] = [3, 3]

#Bloque de funciones:


# Cargar MNIST y dividir en train / test
def cargar_dataset():
	# Carga MNIST
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# Reestructurar a un solo canal
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# Clasificación de target en one-hot
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# Normalización de tonos de pixel
def prep_pixels(train, test):
	# Transformar integers en floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# Normalizar 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# Devolver normalizado
	return train_norm, test_norm

# Graficar un dígito
def graficar_uno(valor, conjunto, prediccion=None):
	fig = plt.figure
	plt.imshow(conjunto[valor], cmap='gray_r')
	if np.array_equiv(conjunto, trainX):
		plt.xlabel('Label train: {}'.format(np.argmax(trainY[valor])))
	if np.array_equiv(conjunto, testX):
		plt.xlabel('Label test: {}'.format(np.argmax(testY[valor])))
	if prediccion is not None:
		plt.ylabel('Predicho: '+str(prediccion))
	return plt.show()

#Esta función permite graficar las matrices de confusión de manera agradable a la vista

def plot_confusion_matrix(y_true, y_pred, classes=np.arange(10),
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Purples):
    if not title:
        if normalize:
            title = 'Matriz de confusión normalizada'
        else:
            title = 'Matriz de confusión sin normalizar'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Matriz de confusión normalizada')
    else:
        print('Matriz de confusión sin normalizar')
    print(cm)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Cifra predicha',
           xlabel='Cifra verdadera')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
np.set_printoptions(precision=3)

# Evaluador SIMPLE

def evaluar_modelo_simple(dataX,dataY,modelo,estadoaleatorio=1,epocas=10):
	scores, histories= list(),list()
	model = modelo()
	history = model.fit(trainX, trainY, epochs=epocas, batch_size=32, validation_data=(testX, testY), verbose=1)
	_, acc = model.evaluate(testX, testY, verbose=1)
	print('> %.3f' % (acc * 100.0))
	scores.append(acc)
	histories.append(history)
	return scores, histories, model

# Evaluación con k-fold cross-validation
def evaluar_modelo_kfold(dataX, dataY, modelo, n_folds=5, estadoaleatorio=1):
	scores, histories = list(), list()
	# preparar los k-fold
	kfold = KFold(n_folds, shuffle=True, random_state=estadoaleatorio)
	# Enumerar las divisiones
	for train_ix, test_ix in kfold.split(dataX):
		# definir model
		model = modelo()
		# elegir filas para train y validation
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fitear modelo
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluar modelo
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# guardar puntajes
		scores.append(acc)
		histories.append(history)
	return scores, histories, model

# Graficar diagnósticos
def diagnosticos(histories):
	for i in range(len(histories)):
		# graficar loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# graficar accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()
trainX, trainY, testX, testY = cargar_dataset()
trainX, testX = prep_pixels(trainX, testX)
def predecirparaunalista(redneuronalentrenada,conjunto=testX):
    puntajes = redneuronalentrenada.predict(conjunto)
    prediccion =np.zeros(len(conjunto),dtype=int)
    for x in range(1,len(conjunto)):
        prediccion[x] = np.argmax(puntajes[x])
    unique, counts = np.unique(prediccion,return_counts=True)
    return dict(zip(unique,counts))
def matrizdeconfusion(redneuronalentrenada,conjunto=testX, targetsonehot=testY, normalizar=False):
    puntajes = redneuronalentrenada.predict(conjunto)
    prediccion =np.zeros(len(conjunto),dtype=int)
    for x in range(1,len(conjunto)):
        prediccion[x] = np.argmax(puntajes[x])
    plot_confusion_matrix(prediccion, np.argmax(targetsonehot, axis= 1), normalize=normalizar, title= "Matriz de confusión")
    plt.show()

def find_non_equal_indices(list1, list2):
    non_equal_indices = []
    for i in range(min(len(list1), len(list2))):
        if list1[i] != list2[i]:
            non_equal_indices.append(i)
    return non_equal_indices

def listarerrores(adivinado,objetivo=testY):
    errores = find_non_equal_indices(adivinado,np.argmax(objetivo, axis = 1))
    return errores
#El MNIST es un conjunto de imágenes de digitos del 0 al 9 escritos a mano. 
# Este dataset tiene mas de 60,000 imágenes separadas en 10 clases. 
# El reto es construir un clasificador de imágenes que sea capaz de reconocer los digitos.
print ("Tamaño de conjunto de entrenamiento: "+str(len(trainX)) + "\n"+"Tamaño de conjunto de test:          "+str(len(testX)))

plt.rcParams['figure.figsize'] = [3, 3]
graficar_uno(315, trainX)
graficar_uno(5626, testX)


### Comencemos entonces de la manera más simple que podamos:
def red_simple():
	model = Sequential() # keras. Agrupa un conjunto de layers dentro de un modelo
	model.add(Flatten()) # Las imágenes son de 28x28, matriciales. Necesitamos achatarlas i.e. 28x28 -> 784 entradas!
	model.add(Dense(1, activation="sigmoid")) # UNA capa oculta con una sola neuronita. Activación más básica que encontramos: Sigmoide.
	model.add(Dense(10, activation='softmax')) # Estamos armando un clasificador... la capa de salida debe ser softmax
	opt = keras.optimizers.SGD(learning_rate=0.01) # Descenso por gradiente. El optimizador más simple
	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy']) #Compilamos las capas secuenciales.
	#                                                                             Como es clasificación, nuestra medida de error es entropía cruzada
    #                                                                             A la vez, queremos que reporte el accuracy
	return model
scores, histories, entrenado = evaluar_modelo_simple(trainX, trainY, red_simple)
#scores, histories, entrenado = evaluar_modelo_kfold(trainX, trainY, red_simple)


# Grafiquemos cómo fue entrenando:

plt.rcParams['figure.figsize'] = [8, 8]
diagnosticos(histories)

#Donde se confunc¿dio la red? Hago la matriz de confusion

matrizdeconfusion(entrenado,normalizar=True)
#Por si queremos evaluar post training:
#entrenado.evaluate(testX,testY)
#Y si entrenamos más épocas?
scores, histories, entrenado = evaluar_modelo_simple(trainX, trainY, red_simple,epocas=50)
diagnosticos(histories)
matrizdeconfusion(entrenado,normalizar=True)

def red_simple2():
	model = Sequential() 
	model.add(Flatten()) 
	model.add(Dense(10, activation="sigmoid")) # UNA capa oculta con DIEZ neuronas. Activación más básica que encontramos: Sigmoide.
	model.add(Dense(10, activation='softmax'))
	opt = keras.optimizers.SGD(learning_rate=0.01) 
	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy']) 
	return model
scores, histories, entrenado = evaluar_modelo_simple(trainX, trainY, red_simple2,epocas=10)
diagnosticos(histories)
matrizdeconfusion(entrenado,normalizar=True)

# y si añado mas capas
def red_simple3():
	model = Sequential() 
	model.add(Flatten()) 
	model.add(Dense(50, activation="sigmoid")) 
	model.add(Dense(20, activation="sigmoid")) 
	model.add(Dense(10, activation='softmax'))
	opt = keras.optimizers.SGD(learning_rate=0.01) 
	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy']) 
	return model
scores, histories, entrenado = evaluar_modelo_simple(trainX, trainY, red_simple3,epocas=50)
diagnosticos(histories)
matrizdeconfusion(entrenado,normalizar=True)

#KFold
scores, histories, entrenado = evaluar_modelo_kfold(trainX, trainY, red_simple)
diagnosticos(histories)
matrizdeconfusion(entrenado,normalizar=True)




#Aquí les dejo algunas redes extra....

def red_teo1():
	model = Sequential()
	model.add(Flatten())
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	opt = keras.optimizers.RMSprop(learning_rate=0.003)
	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def red_teo2():
	model = Sequential()
	model.add(Flatten())
	model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	opt = keras.optimizers.RMSprop(learning_rate=0.003)
	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def red_teo2punto1():
	model = Sequential()
	model.add(Flatten())
	model.add(Dropout(0.0625))
	model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.0625))
	model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	opt = keras.optimizers.Adam(learning_rate=0.002)
	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def red_3():
	model = Sequential()
	model.add(Flatten())
	model.add(Dropout(0.125))
	model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.125))
	model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	opt = keras.optimizers.Adam(learning_rate=0.002)
	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
num = 20
images = trainX[:num]
labels = trainY[:num]

num_row = 4
num_col = 5
# graficar un conjunto
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray_r')
    ax.set_title('Label: {}'.format(np.argmax(labels[i])))
plt.tight_layout()
plt.show()
