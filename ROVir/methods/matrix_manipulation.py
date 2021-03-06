from cv2 import sort
import numpy as np
from numpy import linalg as LA
from scipy.ndimage import gaussian_filter
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize


def calculate_eig(A):
    eigVal, eigVec = LA.eig(A)
    topNv = eigVal.argsort()[::-1]
    # Test to obtain hte bottom coils
    botNv = topNv[-5:]
    #topNv = topNv[:-lowf]
    botweights = eigVec[:, botNv]
    return topNv, eigVec, botweights


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


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w


def generate_matrix(coils):
    """Generate a matrix according to ROVir method"""

    ncoils = coils.shape[-1]
    matrix = np.zeros((ncoils, ncoils))

    for i in range(ncoils):
        for j in range(ncoils):
            matrix[i, j] = np.sum(coils[:, j].T.dot(coils[:, i]))

    return matrix


def filter_coils(coils, sigma=25):
    """Apply an extreme gaussian filter to all coils

    Going through each coil, we apply a strong gaussian filter
    to it, so that we can find the weight of the signal at 
    that coil
    """

    new_coils = np.zeros(coils.shape)
    for i in range(coils.shape[-1]):
        new_coils[..., i] = normalize_matrix(gaussian_filter(coils[..., i],
                                                             sigma=sigma))

    return new_coils


def generate_virtual_coils(coils, weights, topNv):
    v_coils = np.zeros(coils.shape)
    ncoils = coils.shape[-1]
    for j in range(topNv):
        total = 0
        for l in range(ncoils):
            try:
                total += weights[l, j]*coils[:, :, l]
            except:
                pass
        v_coils[:, :, j] = total

    return v_coils


def normalize_matrix(matrix):
    return (matrix-np.min(matrix))/np.max(matrix)


def expand_weights(weights, size):
    weights_ = np.ones(size)
    weights_[:weights.shape[0], :weights.shape[1]] = weights
    return np.absolute(weights_)
