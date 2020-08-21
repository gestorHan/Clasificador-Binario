import numpy as np
import itertools

def generar_lambdas(level, m_parametros):
    arr_range = np.arange(m_parametros)
    rng = np.random.default_rng()
    lista_combinaciones = list(set(itertools.combinations(arr_range ,level)))
    if (level == 1):
        return lista_combinaciones
    rng.shuffle(lista_combinaciones)    
    print(lista_combinaciones[:m_parametros])
    return lista_combinaciones[:m_parametros]

#def vecto_t ()    

def generar_tabla(bin_matrix_Q, labels, lambda_actual):
    print ("Lambda actual : " , lambda_actual)
    for row_i in lambda_actual:
        print("Obtener la fila :" , row_i) 
       
    return []

if __name__ == "__main__":

    bin_matrix_Q = np.array(
        [
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, 1, ],
            [-1, 1, -1, -1, -1, -1, -1, -1, -1, 1, ],
            [-1, -1, 1, -1, -1, -1, -1, -1, -1, 1, ],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, ],
        ]
    )
    labels = np.array(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ]
    )
    m_parametros, n_puntos =  bin_matrix_Q.shape
    print(m_parametros)
    print(n_puntos)
    arreglo_lambdas = []
    lambda_actual = []
    tabla_actual = []
    for level in range(1, 5):
        arreglo_lambdas.append(generar_lambdas(level, m_parametros))
        lambdas_actuales = arreglo_lambdas[level-1]
        for lambda_actual in lambdas_actuales:
            tabla_actual = generar_tabla(bin_matrix_Q,labels,lambda_actual)
    print(arreglo_lambdas)
