#Instalar scikit-learn: pip install -U scikit-learn
#Instalar matplotlib: pip install matplotlib
#Actualizar ver 20.2.2: python.exe -m pip install --upgrade pip
import time
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

train_samples = 5000#Tamanio de muestras
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)#Obtener el conjunto de datos de openml por nombre o id del conjunto de datos.
#X->data, y->target
random_state = check_random_state(0)#Elige una posicion aleatoria
permutation = random_state.permutation(X.shape[0])#Paso la imagen de la posicion aleatoria
#.permutation(X.shape[0]) permuta elementos entre 0 y la longitud de la fila
#X.shape[0] proporciona la longitud de la primera fila
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))#Acomoda la matriz X en X.shape[0] filas y 2 columnas, el -1(Valor no especificado) toma valor 2

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=5000)

'''
X viene a ser X_train, y viene a ser X_test, y_train=representa el
número absoluto de muestras de entrenamiento, y_test = (Numero absoluto de las pruebas)
'''


num = 60
images = X_train [: num] 
labels = y_train [: num]
'''
for i in range(60):
    l1_plot = plt.subplot(6, 10, i + 1)#Num de filas,Num de columnas, indice 
    l1_plot.imshow(images[i].reshape(28, 28),cmap=plt.cm.gray)

plt.show()
'''

print(images.shape)
#PASAR EL ARRELGO TRIDIMENSIONAL A BIDIMENSIONAL
#images es de 60x(28*28)
'''
for i in range(300,400):
    print(images.item(25,i))
'''
images = np.transpose(images)

#images es de (784)x60
print(images.shape)
'''
for i in range(300,400):
    print(images[i][25])
'''
#Creo un arreglo de ceros [784]
median = np.zeros(784)
t=0
#sumo toda la fila de mis elementos a cada posicion
for i in range(784):
    for j in range(60):
        median[t]=median[t]+images[i][j]
    t=t+1
'''
for i in range(300,400):
    print(images[i][25])
'''
#saco la media a cada suma hallada
for i in range(784):
    median[i]=median[i]/60

'''
for i in range(784):
    print(median[i])
'''
#vamos a restar a cada valor la media en la fila correspondiente:
for i in range(784):
    for j in range(60):
        images[i][j] = images[i][j] - median[i]
'''
for i in range(300,400):
    print(images[i][25])
'''

#Generar matriz Gaussina de 10x784
mu, sigma = 0, 1 # mean and standard deviation

A = np.zeros((10, 784))
for i in range(10):
    s = np.random.normal(mu, sigma, 784)
    for j in range(784):
        A[i][j] = A[i][j] +s[j]

#Hallaremos Q = A*X
#Multiplicamos A*X
Q = np.dot(A,images)

#Aplicaremos la funcion Signo a toda la matriz Q:
for i in range(10):
    for j in range(60):
        if (Q[i][j]<0):
            Q[i][j]=-1
        if(0<=Q[i][j]):
            Q[i][j]=1
print(Q)