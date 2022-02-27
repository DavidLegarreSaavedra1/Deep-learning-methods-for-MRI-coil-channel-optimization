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
    vec = matrix.flatten().reshape(
        rows*cols, dim
    )
    return vec, [rows, cols]


def vec_to_matrix(vec, rows, cols):

    return vec.reshape(
        rows, cols, vec.shape[1]
    )


def generate_matrix(imgs):
    matrix = np.zeros((imgs.shape[1], imgs.shape[1]))
    for i in range(imgs.shape[1]):
        for j in range(imgs.shape[1]):
            matrix[i, j] += imgs[:, i].T.dot(imgs[:, j])

    return matrix


def filter_coils(coils):
    new_coils = np.zeros(coils.shape)
    for i in range(coils.shape[2]):
        new_coils[:, :, i] = LA.norm(
            gaussian_filter(coils[:, :, i], sigma=5))

    return new_coils


def generate_virtual_coils(coils, weights):
    v_coils = np.zeros(coils.shape)
    print(f"{v_coils.shape=}")
    for coil in range(coils.shape[1]):
        v_coils[:, coil] = weights[coil]*coils[:, coil]
    print(f"{v_coils.shape=}")

    return v_coils
