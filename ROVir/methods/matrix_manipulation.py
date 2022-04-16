import numpy as np
from numpy import linalg as LA
from scipy.ndimage import gaussian_filter
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize


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
    matrix = np.zeros((ncoils, ncoils)).astype(np.csingle)
    for i in range(ncoils):
        for j in range(ncoils):
            matrix[i, j] = np.sum(
                np.matrix(coils[i, :, :]).H.dot(coils[i, :, :]))

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
                                                             sigma=20))

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
    return matrix/np.max(matrix)


def expand_weights(weights, size):
    weights_ = np.ones(size)
    weights_[:weights.shape[0], :weights.shape[1]] = weights
    return np.absolute(weights_)


def gram_schmidt(A):

    (n, m) = A.shape

    for i in range(m):

        q = A[:, i]  # i-th column of A

        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]

        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError(
                "The column vectors are not linearly independent")

        # normalize q
        q = q / np.sqrt(np.dot(q, q))

        # write the vector back in the matrix
        A[:, i] = q

    return A
