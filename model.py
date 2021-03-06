import numpy as np
import itertools

class model:
    def __init__ (self , max_level):
        self.parametros = []
        self.level =  max_level

    def agregar_tupla (self , lamb:list ,tabla):
        self.parametros.append ( (lamb , tabla))

    def mostrar_parametros (self):
        print ("Parametros:" , self.parametros)
        print ("Nivel maximo :" , self.level)
    
    def clasificar (self,nuevo : list):
        puntaje_etiquetas = np.zeros(2)
        for parametro in self.parametros:
            actual_lambda , actual_tabla = parametro
            sub_nuevo = []
            for value in actual_lambda:
                sub_nuevo.append(nuevo[value])
            identificador_t = bin_to_number(sub_nuevo)
            puntaje_etiquetas += actual_tabla.T[identificador_t]
        print (10*"***")
        print ("Puntaje:")
        print (puntaje_etiquetas)
        print (10*"***")

def generar_lambdas(level, m_parametros):
    arr_range = np.arange(m_parametros)
    rng = np.random.default_rng()
    lista_combinaciones = list(set(itertools.combinations(arr_range ,level)))
    if (level == 1):
        return lista_combinaciones
    rng.shuffle(lista_combinaciones)    
    print(lista_combinaciones[:m_parametros])
    return lista_combinaciones[:m_parametros]

def bin_to_number( bin_vector):
    bin_vector = np.array(bin_vector)
    bin_vector = np.where (bin_vector == 1, bin_vector, 0 ) 
    t = 0
    for i,j in enumerate (bin_vector):
        t += j <<i
    print ("T de : ", bin_vector)
    print ("T:" , t)
    return t

def generar_vector_t (submatrix_lambda:list):
    vector_t = []
    submatrix_lambda = np.array(submatrix_lambda).T
    for column in submatrix_lambda:
        vector_t.append (bin_to_number(column))
    return vector_t


def tablas_conteo(vector_t, labels, level):
    pares_unicos, conteo = np.unique(
        np.array([vector_t, labels]).T
        , axis= 0, return_counts=True
    )
    tablas = np.array([
        np.zeros(2**level),
        np.zeros(2**level)
    ])

    print("Pares unicos:" , pares_unicos)
    print("Conteo:", conteo)
    
    for par_unico, conteo_actual in zip(pares_unicos, conteo):
        tablas[par_unico[1]][par_unico[0]] = int(conteo_actual)
    return tablas


def tabla_indices_de_miembro(tabla_conteo , suma_conteo , level):
    tabla_indices = np.array([
        np.zeros(2**level),
        np.zeros(2**level)
    ])

    for n_row , row in enumerate(tabla_conteo):
        for n_col , elem in enumerate (row):
            if (elem != 0):
                tabla_indices[n_row][n_col] = (elem * (np.abs(suma_conteo[n_col]-2*elem))) /(suma_conteo[n_col]**2)
    return tabla_indices

def generar_tabla(bin_matrix_Q, labels, lambda_actual):
    submatrix_lambda = []
    vector_t = []
    print ("Lambda actual : " , lambda_actual)
    for row_i in lambda_actual:
        print("Obtener la fila :" , row_i) 
        submatrix_lambda.append(bin_matrix_Q[row_i])
    vector_t = generar_vector_t(submatrix_lambda)
    tabla_conteo = tablas_conteo(vector_t , labels , len(lambda_actual))
    suma_conteo = np.sum(tabla_conteo , axis=0)
    tabla_indices = tabla_indices_de_miembro (tabla_conteo , suma_conteo , len(lambda_actual))
    
    print ("Lambda actual :\n" , lambda_actual  ," size: " ,len(lambda_actual) )
    print ("Lambda lista actual :\n" ,list(lambda_actual)  ," size: " ,len(lambda_actual) )
    print ("Tabla de conteo:\n" , tabla_conteo)
    print ("Suma de columnas:\n", suma_conteo)
    print ("Tabla de indices de miembro \n" , tabla_indices)
    return tabla_indices

def entrenar_modelo (bin_matriz_Q ,labels ,max_level, types_labels=2):
    modelo = model(max_level)
    arreglo_lambdas = []
    lambda_actual = []

    m_parametros, n_puntos =  bin_matrix_Q.shape
    print(m_parametros)
    print(n_puntos)

    for level in range (1, max_level+1):
        arreglo_lambdas.append(list(generar_lambdas(level, m_parametros)))
        lambdas_actuales = arreglo_lambdas[level-1]
        for lambda_actual in lambdas_actuales:
            modelo.agregar_tupla(
                lambda_actual,
                generar_tabla(bin_matrix_Q,labels,lambda_actual)
                )
    return modelo

if __name__ == "__main__":

    levels = 3 

    bin_matrix_Q = np.array(
        [
            [-1, 1, 1, -1, -1, -1, -1, 1, -1, 1, ],
            [-1, 1, -1, -1, 1, -1, 1, -1, 1, 1, ],
            [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1, ],
            [-1, 1, 1, -1, -1, -1, -1, 1, -1, -1, ],
        ]
    )
    labels = np.array(
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, ]
    )

    mi_modelo = entrenar_modelo(bin_matrix_Q , labels ,levels)
    mi_modelo.mostrar_parametros()
    mi_modelo.clasificar( [1,1,1,1])