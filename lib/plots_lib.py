import numpy as np
import matplotlib.pyplot as plt
import os


def subplots1D(d, keymat):
    dim = keymat.shape
    fig, ax = plt.subplots(nrows=dim[0], ncols=dim[1], figsize=(dim[0]*3, dim[1]*3))
    for i in range(dim[0]):
        for j in range(dim[1]):
            for key in keymat[i][j]:
                ax[i][j].plot(d[key], label=key)
            ax[i][j].legend()
plt.show()


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
