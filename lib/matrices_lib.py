import random
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

# Uniformly distributed random matrix 
def randUMat(dim, rng):
    return rng[0] + (rng[1] - rng[0]) * np.random.rand(dim[0], dim[1])


# Clip all elements of matrix
def clipMat(M, rng):
    M[M < rng[0]] = rng[0]
    M[M > rng[1]] = rng[1]
    return M


# Calclate L1 norm of a sparse matrix or numpy array
def L1Avg(M):
    if type(M) is np.ndarray:
        return np.average(np.abs(M)) if len(M) > 0 else 0.0
    elif type(self.v[key]) is scipy.sparse.csr.csr_matrix:
        return L1Avg(M.data)
    else:
        raise ValueError('Unexpected Variable Data Type: ', type(M))

        
# Uniform-random matrix, with given uniform-random sparsity
def spRandUMatBase(dim, rng, p_sp, withDiag=True):
#     # Compute number of sparse elements
#     size1D = dim[0] * dim[1]
#     size1Dsparse = int(size1D * p_sp)
    
    # Generate all possible indices. User may want to explicitly exclude off-diagonal.
    if withDiag:
        idx2Dflat = np.array([(i, j) for i in range(dim[0]) for j in range(dim[1])]).transpose()
    else:
        idx2Dflat = np.array([(i, j) for i in range(dim[0]) for j in range(dim[1]) if i!=j]).transpose()
    
    # Randomly sample exactly the fraction p_sp of all available connections
    pConn2Dflat = np.random.uniform(0, 1, idx2Dflat.shape[1])
    conn2Dflat = pConn2Dflat <= np.quantile(pConn2Dflat, p_sp)
    
    idx2Drow = idx2Dflat[0, conn2Dflat]
    idx2Dcol = idx2Dflat[1, conn2Dflat]
    
    # Generate random values
    val = np.random.uniform(rng[0], rng[1], idx2Drow.shape[0])  
    
#     # Generate locations of non-zero matrix elements, uniformly distributed over the matrix
#     #idx1D = np.random.permutation(size1D)[:size1Dsparse]  # NOTE: this is too memory-hungry for large sizes
#     idx1D = np.array(random.sample(range(0, size1D), size1Dsparse))
    
#     idx2Drow = idx1D // dim[1]
#     idx2Dcol = idx1D % dim[1]
    
    return val, (idx2Drow, idx2Dcol)


# Sparse uniformly-distributed random matrix
def spRandUMat(dim, rng, p_sp, dtype=float, withDiag=True):
    val, idx = spRandUMatBase(dim, rng, p_sp, withDiag=withDiag)
    return csr_matrix((val, idx), shape=dim, dtype=dtype)

# Sparse uniformly-distributed random matrix
def spRandUMatFromBase(base, dim, dtype=float):
    return csr_matrix(base, shape=dim, dtype=dtype)
    
# Takes several matrix bases and merges them into one matrix base
# All matrix bases are given in their own coordinates, so need to be shifted to be placed correctly into the large base
# It is user's responsibility that shifts make sense and matrices don't overlap
def mergeMatBases(bases, rowShifts, colShifts):
    vals = []
    idxRowShifted = []
    idxColShifted = []
    
    for base, rowSh, colSh in zip(bases, rowShifts, colShifts):
        vals += [base[0]]
        idxRowShifted += [base[1][0] + rowSh]
        idxColShifted += [base[1][1] + colSh]
        
    return np.hstack(vals), (np.hstack(idxRowShifted), np.hstack(idxColShifted))
    
# # Matrix-vector dot-product for base matrix.
# # Since matrix values need to be edited, it is cheaper to never convert matrix base to actual matrix, but instead do operations on the base directly
# def baseMV(base, v, dimBase):
#     rez = np.zeros(dimBase[0])
#     for val, iRow, iCol in zip(base[0], base[1][0], base[1][1]):
#         rez[iRow] += val*v[iCol]
#     return rez

# Calculate outer product of vectors sparcity indices
# Returns a 1D flat vector corresponding to values of resulting sparse matrix
def outerSP(idxRow, idxCol, vrow, vcol):
    return np.multiply(vrow[idxRow], vcol[idxCol])