import numpy as np
from numpy import linalg as LA
from scipy.ndimage import gaussian_filter
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def calculate_eig(A, lowf):
    eigVal, eigVec = LA.eig(A)
    topNv = eigVal.argsort()[::-1]
    topNv = topNv[:-lowf]
    eigVal = eigVal[topNv]
    eigVec = eigVec[:, topNv]
    return topNv, eigVal, eigVec


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
            matrix[i, j] = np.sum(coils[:, i].T.dot(coils[:, j]))

    return matrix


def generate_matrix_im(coils):
    """Generate a matrix according to ROVir method"""

    ncoils = coils.shape[0]
    matrix = np.zeros((ncoils, ncoils))
    for i in range(ncoils):
        for j in range(ncoils):
            matrix[i, j] = np.sum(np.matrix(coils[i,:,:]).H.dot(coils[i,:,:]))

    return matrix


def filter_coils(coils):
    """Apply an extreme gaussian filter to all coils

    Going through each coil, we apply a strong gaussian filter
    to it, so that we can find the weight of the signal at 
    that coil
    """

    new_coils = np.zeros(coils.shape)
    for i in range(coils.shape[-1]):
        new_coils[..., i] = normalize_matrix(gaussian_filter(coils[..., i],
                                                             sigma=1))

    return new_coils


def generate_virtual_coils(coils, weights, topNv):
    v_coils = np.zeros(coils.shape)
    ncoils = coils.shape[-1]

    for j in range(topNv):
        total = 0
        for l in range(ncoils):
            total += weights[l, j]*coils[:, :, l]
        v_coils[:, :, j] = total

    return v_coils


def plot_coils(coils, title=''):
    ncoils = coils.shape[-1]
    x = int(np.floor(np.sqrt(ncoils)))
    fig, axs = plt.subplots(x, x)
    i = 0
    for ax in axs.reshape(ncoils):
        ax.imshow(coils[..., i], cmap='gray',  extent=[0, 1, 0, 1])
        i += 1
    fig.suptitle(title, fontsize=16)
    return


def normalize_matrix(matrix):
    return matrix/LA.norm(matrix)


def expand_weights(weights, size):
    weights_ = np.zeros(size)
    weights_[:weights.shape[0], :weights.shape[1]] = weights
    return np.absolute(weights_)

