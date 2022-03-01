import numpy as np
import sys
from numpy import linalg as LA, vectorize
from methods.matrix_manipulation import *
from methods.combine_images import *
import matplotlib.pyplot as plt


def ROVir(coils, regions, lowf):
    A_W, A_H, B1_W, B1_H, B2_W = regions

    w = filter_coils(coils)

    HEIGHT, WIDTH, NCOILS = w.shape

    A = np.zeros(w.shape)
    B = np.zeros(w.shape)

    A[A_H, A_W, :] = w[A_H, A_W, :]
    B[B1_H, B1_W, :] = w[B1_H, B1_W, :]
    B[B1_H, B2_W, :] = w[B1_H, B2_W, :]

    # Convert regions to vectors for ease of calculation
    A = matrix_to_vec(A)
    B = matrix_to_vec(B)

    # Generate the regions according to ROVir method
    A = generate_matrix(A)
    B = generate_matrix(B)
    
    print(f"{A.shape=}")
    
    # Transform A and B back to matrices to plot
    #A = vec_to_matrix(A, HEIGHT, WIDTH)
    #B = vec_to_matrix(B, HEIGHT, WIDTH)

    comb = LA.inv(B).dot(A)
    topNv, eigVec = calculate_eig(comb, lowf)
    weights = np.real_if_close(eigVec, tol=1)
    
    print(f"{w.shape=}")
    print(f"{weights.shape=}")
    
    #v_coils = generate_virtual_coils(w, weights)
    
    plot_coils(w)
    
    _ = plt.imshow(comb, cmap='gray')
    plt.show()
    return w
