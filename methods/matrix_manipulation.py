import numpy as np
from numpy import linalg as LA
from scipy.ndimage import gaussian_filter
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def calculate_eig(A, lowf):
    eigenValues, eigenVectors = LA.eig(A)
    idx = eigenValues.argsort()[::-1]
    idx = idx[:-lowf]
    eigVal = eigenValues[idx]
    eigVec = eigenVectors[:,idx]
    print(f"{eigVal.shape=}")
    print(f"{eigVec.shape=}")
    
    return idx, eigVec


def matrix_to_vec(matrix):
    rows, cols, dim = matrix.shape
    vec = matrix.flatten().reshape(
        rows*cols, dim
    )
    return vec


def vec_to_matrix(vec, rows, cols):
    matrix = vec.reshape(
        rows, cols, vec.shape[1]
    )

    return matrix


def generate_matrix(coils):
    """Generate a matrix according to ROVir method"""

    ncoils = coils.shape[-1]
    matrix = np.zeros((ncoils, ncoils))
    for i in range(ncoils):
        for j in range(ncoils):
            matrix[i,j] = np.sum(coils[:,i].T.dot(coils[:,j]))

    return matrix 



def filter_coils(coils):
    """Apply an extreme gaussian filter to all coils

    Going through each coil, we apply a strong gaussian filter
    to it, so that we can find the weight of the signal at 
    that coil
    """

    new_coils = np.zeros(coils.shape)
    for i in range(coils.shape[2]):
        new_coils[:, :, i] = normalize_matrix(gaussian_filter(coils[:, :, i],
                                              sigma=50))

    return new_coils


def generate_virtual_coils(coils, weights):
    v_coils = np.zeros(coils.shape)
    ncoils = coils.shape[-1]
    
    for i in range(ncoils):
        for j in range(ncoils):
            j_weight = np.multiply(np.ones(v_coils.shape[:2]),weights[:,j])
            v_coils[:,:,i] += j_weight*coils[:,:,j]

    return v_coils


def plot_coils(coils):
    ncoils = coils.shape[-1]
    x = int(np.floor(np.sqrt(ncoils)))
    fig, axs = plt.subplots(x,x)
    i = 0
    for ax in axs.reshape(ncoils):
        ax.imshow(coils[...,i], cmap='gray')
        i += 1
    plt.show()
    return

def normalize_matrix(matrix):
    max_pixel = matrix.max()

    return matrix * 1/max_pixel
