import numpy as np
import sys
from numpy import linalg as LA, vectorize
from .matrix_manipulation import *
from .image_manipulation import *
import matplotlib.pyplot as plt


def ROVir(coils, regions, lowf=0):
    A_W, A_H, B1_W, B1_H, B2_W = regions

    print("Filtering coils...")
    #w_coils = coils*filter_coils(coils)
    w_coils = filter_coils(coils, 12)

    #plot_coils(w_coils)
    HEIGHT, WIDTH, NCOILS = w_coils.shape
    #plot_coils(w_coils, 'W_coils')

    A = np.zeros(w_coils.shape)
    B = np.zeros(w_coils.shape)

    A[A_H, A_W, :] = w_coils[A_H, A_W, :]
    B[B1_H, B1_W, :] = w_coils[B1_H, B1_W, :]
    B[B1_H, B2_W, :] = w_coils[B1_H, B2_W, :]

    coil = 2
    fig, ax = plt.subplots()
    ax.imshow(w_coils[..., coil])
    fig, ax = plt.subplots()
    ax.imshow(coils[...,coil])
    

    plot_masks(
        combine_images(coils),
        combine_images(A),
        combine_images(B),
        save=True
    )
    
    plot_image(
        combine_images(w_coils)
    )
    
    # Convert regions to vectors for ease of calculation
    A = matrix_to_vec(A)
    B = matrix_to_vec(B)

    # Generate the regions according to ROVir method
    print("Generating matrices...")
    A = generate_matrix(A)
    B = generate_matrix(B)

    comb = LA.inv(B)*A

    topNv, topweights, botweights = calculate_eig(comb)

    v_coils = generate_virtual_coils(coils, topweights, len(topNv))

    bot_v_coils = generate_virtual_coils(coils, botweights, 25-len(topNv))

    if lowf!=0:
        v_coils = v_coils[:,:,:-lowf]

    return v_coils, bot_v_coils
    # return w_coils
