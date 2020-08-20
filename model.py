import numpy as np
import itertools
#m <- numero de caracteristicas
#l <- nivel actual
import itertools
def findsubsets(shuffled,l , m):
    return list(set(itertools.combinations(shuffled, l)))[:m]

def generar_lambdas(l,m):
    shuffled = np.arange(m)
    if l == 1:
        return list(set(itertools.combinations(shuffled,1)))
    rng = np.random.default_rng()
    rng.shuffle(shuffled)
    print (shuffled)
    return findsubsets (shuffled,l,m)

if __name__ == "__main__":

    Q = np.array(
        [
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, 1, ],
            [-1, 1, -1, -1, -1, -1, -1, -1, -1, 1, ],
            [-1, -1, 1, -1, -1, -1, -1, -1, -1, 1, ],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, ],
        ]
    )
    m , n = Q.shape
    print (m)
    print (n)
    print(generar_lambdas(2,30))
    