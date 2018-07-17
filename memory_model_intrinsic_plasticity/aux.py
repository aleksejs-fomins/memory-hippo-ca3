import random
import numpy as np
from scipy.sparse import csr_matrix

def randUMat(dim, rng):
    return rng[0] + (rng[1] - rng[0]) * np.random.rand(dim[0], dim[1])

def clipMat(M, rng):
    M[M < rng[0]] = rng[0]
    M[M > rng[1]] = rng[1]
    return M

# Uniform-random matrix, with given uniform-random sparsity
def spRandUMat(dim, rng, p_sp):
    # Compute number of sparse elements
    size1D = dim[0] * dim[1]
    size1Dsparse = int(size1D * p_sp)
    
    # Generate random values
    val = np.random.uniform(rng[0], rng[1], size1Dsparse)
    
    # Generate locations of non-zero matrix elements, uniformly distributed over the matrix
    #idx1D = np.random.permutation(size1D)[:size1Dsparse]  # NOTE: this is too memory-hungry for large sizes
    idx1D = np.array(random.sample(range(0, size1D), size1Dsparse))
    
    idx2Drow = idx1D / dim[1]
    idx2Dcol = idx1D % dim[1]
    
    return csr_matrix((val, (idx2Drow, idx2Dcol)), shape=dim)
    
    