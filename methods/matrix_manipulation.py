import numpy as np
from numpy import linalg as LA
from scipy.ndimage import gaussian_filter
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def calculate_eig(A, lowf):
    eigenValues, _ = LA.eig(A)
    idx = eigenValues.argsort()[::-1]
    idx = idx[:-lowf]
    return idx


def matrix_to_vec(matrix):
    rows, cols, dim = matrix.shape
    print(f"{rows,cols,dim=}")
    vec = matrix.flatten().reshape(
        rows*cols, dim
    )
    print(f"{vec.shape=}")
    return vec


def vec_to_matrix(vec, rows, cols):
    matrix = vec.reshape(
        rows, cols, vec.shape[1]
    )

    return matrix


def generate_matrix(coils):
    """Generate a matrix according to ROVir method"""

    new_coils = np.zeros(coils.shape)
    for i in range(coils.shape[-1]):
        new_coils[:, :, i] = coils[:, :, i].T.dot(coils[:, :, i])

    print(f"{new_coils.shape=}")
    return np.sum(new_coils, axis=2)


def filter_coils(coils):
    """Apply an extreme gaussian filter to all coils

    Going through each coil, we apply a strong gaussian filter
    to it, so that we can find the weight of the signal at 
    that coil
    """

    new_coils = np.zeros(coils.shape)
    for i in range(coils.shape[2]):
        new_coils[:, :, i] = LA.norm(
            gaussian_filter(coils[:, :, i], sigma=50))

    return new_coils


def generate_virtual_coils(coils, weights):
    v_coils = np.zeros(coils.shape)
    print(f"{v_coils.shape=}")
    for coil in range(coils.shape[1]):
        v_coils[:, coil] = weights[coil]*coils[:, coil]
    print(f"{v_coils.shape=}")

    return v_coils
