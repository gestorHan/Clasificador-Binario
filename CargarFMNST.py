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

train_samples = 5000
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)#Obtener el conjunto de datos de openml por nombre o id del conjunto de datos.
#X->data, y->target
random_state = check_random_state(0)#Elige una posicion aleatoria
permutation = random_state.permutation(X.shape[0])#Paso la imagen de la posicion aleatoria
#.permutation(X.shape[0]) permuta elementos entre 0 y la longitud de la fila
#X.shape[0] proporciona la longitud de la primera fila
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))#Acomoda la matriz X en X.shape[0] filas y 2 columnas, el -1(Valor no especificado) toma valor 2

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=5000)
num = 100
images = X_train [: num] 
labels = y_train [: num]

for i in range(100):
    l1_plot = plt.subplot(10, 10, i + 1)#Num de filas,Num de columnas, indice 
    l1_plot.imshow(images[i].reshape(28, 28),cmap=plt.cm.gray)

plt.show()
