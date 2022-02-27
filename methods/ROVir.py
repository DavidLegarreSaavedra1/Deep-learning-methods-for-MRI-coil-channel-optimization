import numpy as np
import sys
from numpy import linalg as LA
from methods.matrix_manipulation import *
import matplotlib.pyplot as plt


def ROVir(coils, regions, lowf):
    A_W, A_H, B1_W, B1_H, B2_W = regions

    w = filter_coils(coils)

    A = np.zeros(w.shape)
    B = np.zeros(w.shape)

    A[A_H, A_W, :] = w[A_H, A_W, :]
    B[B1_H, B1_W, :] = w[B1_H, B1_W, :]
    B[B1_H, B2_W, :] = w[B1_H, B2_W, :]

    return w
