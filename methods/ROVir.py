import numpy as np
from numpy import linalg as LA
from scipy.ndimage import gaussian_filter
from scipy import signal


def ROVir(coils, regions, lowf):
    A_W, A_H, B1_W, B1_H, B2_W = regions

    new_image = np.zeros(coils.shape)
    w = filter_coils(coils)
    tmp_A = w[A_H, A_W, :]
    img_A = tmp_A.flatten().reshape(tmp_A.shape[0]*tmp_A.shape[1], w.shape[2])

    tmp_B1 = w[B1_H, B1_W, :]
    tmp_B2 = w[B1_H, B2_W, :]

    img_B1 = tmp_B1.flatten().reshape(
        tmp_B1.shape[0]*tmp_B1.shape[1], w.shape[2])
    img_B2 = tmp_B2.flatten().reshape(
        tmp_B2.shape[0]*tmp_B2.shape[1], w.shape[2])

    img_B = np.append(img_B1, img_B2)
    print(f"{img_B.shape=}")

    A = generate_matrix(img_A)
    B = generate_matrix(img_B)

    # Calculate eigenvalues and eigenvectors both matrices
    general_matrix = LA.inv(B).dot(A)
    weights = calculate_eig(general_matrix, lowf)
    print(f"{weights=}")

    new_image = coils[:, :, weights]
    print(f"{new_image.shape=}")

    return new_image


def calculate_eig(A, lowf):
    eigenValues, _ = LA.eig(A)
    idx = eigenValues.argsort()[::-1]
    print(f"{eigenValues=}")
    idx = idx[:-lowf]
    return idx


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


def join_matrices_padding(A, B, p=0):
    C = np.ones((A.shape[0]+B.shape[0], A.shape[1] + B.shape[1], A.shape[2]))*p
    C[:A.shape[0], :A.shape[1], :] = A
    C[A.shape[0]:, A.shape[1]:, :] = B

    return C
