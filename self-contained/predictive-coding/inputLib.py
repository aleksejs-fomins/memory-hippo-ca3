import numpy as np

# Generate a bunch on random images using truncated 2D fourier basis with random coefficients
# All pixel values are non-negative
def genRandomImg(NX, NY, NTERMS=2):
    img = np.zeros((NX, NY))
    xx = np.array([[i for j in range(NY)] for i in range(NX)])
    yy = np.array([[j for j in range(NY)] for i in range(NX)])
    for i in range(-NTERMS, NTERMS+1):
        for j in range(-NTERMS, NTERMS+1):
            kx = i / NX / 2
            ky = j / NY / 2
            r0, phi0 = np.random.uniform(0, 1, 2)
            phi = kx*xx + ky*yy - phi0
            img += r0 * np.sin(2 * np.pi * phi)
    
    img -= np.min(img)
    img /= np.linalg.norm(img)
    return img