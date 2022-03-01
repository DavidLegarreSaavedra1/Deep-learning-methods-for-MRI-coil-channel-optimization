import numpy as np
import sys
from numpy import linalg as LA, vectorize
from methods.matrix_manipulation import *
from methods.combine_images import *
import matplotlib.pyplot as plt


def ROVir(coils, regions, lowf):
    A_W, A_H, B1_W, B1_H, B2_W = regions

    w_coils = coils*filter_coils(coils)

    HEIGHT, WIDTH, NCOILS = w_coils.shape

    A = np.zeros(w_coils.shape)
    B = np.zeros(w_coils.shape)

    A[A_H, A_W, :] = w_coils[A_H, A_W, :]
    B[B1_H, B1_W, :] = w_coils[B1_H, B1_W, :]
    B[B1_H, B2_W, :] = w_coils[B1_H, B2_W, :]

    # Convert regions to vectors for ease of calculation
    A = matrix_to_vec(A)
    B = matrix_to_vec(B)

    # Generate the regions according to ROVir method
    A = generate_matrix(A)
    B = generate_matrix(B)

    comb = LA.inv(B).dot(A)
    topNv, eigVal, weights = calculate_eig(comb, lowf)

    #weights = expand_weights(eigVec, (NCOILS, NCOILS))

    print(f'{w_coils.shape=}')
    print(f'{weights=}')

    v_coils = generate_virtual_coils(coils, weights, len(topNv))

    return v_coils
