from ast import Slice
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import h5py
from pyparsing import line
from methods import *
from methods import ROVIR_im
import math

dirs = [os.getcwd(), "data"]
data_path = os.path.join(*dirs)

A_W = slice(100, 420)
B1_W = slice(0, 90)
B2_W = slice(450, -1)
A_H = slice(90, 440)
B1_H = slice(90, 440)

LINE = 300
DEBUGGING = False

regions = [A_W, A_H, B1_W, B1_H, B2_W]


if __name__ == '__main__':
    img = nib.load(os.path.join(data_path, "Slice44-AllChannels.nii"))

    height = img.shape[0]
    width = img.shape[1]
    num_coils = img.shape[2]
    # lowf = int(input("lowf = "))
    lowf = 8

    img_np = np.array(img.dataobj)

    prev_img = combine_images(img_np)

    fig, axs = plt.subplots(1, 2)
    regions = [A_W, A_H, B1_W, B1_H, B2_W]

    rovir_coils = ROVir(img_np, regions,  lowf)

    new_img = combine_images(rovir_coils)

    plot_images(prev_img,
                "Before ROVir",
                new_img,
                "After ROVir"
                )

    intensity_plot(prev_img, 270, 'Before ROVir')
    intensity_plot(new_img, 270, 'After ROVir')
    plt.show()
