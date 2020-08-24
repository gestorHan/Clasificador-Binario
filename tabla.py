import numpy as np
import itertools


def tabla_conteo(vector_t, labels):
    pares = np.array([vector_t, labels])
    pares_unicos = np.unique(pares.T, axis=0, return_counts=True)
    print("El vector t:", np.array(vector_t))
    print("El vector b:", labels)
    #print ("Conteo ? : " , pares_unicos)
    return pares_unicos


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


if __name__ == "__main__":
    t = [0, 4, 5, 4, 2, 4, 2, 4, 2, 7]
    b = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    print ("Tablas :\n" , tablas_conteo(t , b, 3))


    pass
