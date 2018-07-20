import random
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os

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
    

def saveMatPlot(mat, filename, dpi):
    fig = plt.figure(figsize=(mat.shape[0]/dpi, mat.shape[1]/dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(mat.T)
    plt.savefig(filename)
    plt.close()
    
    
def longMatMovie(mat, window, filebasename, dpi, fps):
    fig = plt.figure(figsize=(window/dpi, mat.shape[1]/dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    thisPlot = plt.imshow(mat[:window].T)
    plt.savefig(filebasename+"0.png")
    
    for i in range(1, len(mat) - window):
        thisPlot.set_data(mat[i:i+window].T)
        plt.savefig(filebasename+str(i)+".png")
        #saveMatPlot(mat[i:i+window], filebasename+str(i)+".png", dpi)
    
    plt.close()
    
    os.system("ffmpeg -r "+str(fps)+" -i "+filebasename+"%01d.png -vcodec mpeg4 -y movie.mp4")