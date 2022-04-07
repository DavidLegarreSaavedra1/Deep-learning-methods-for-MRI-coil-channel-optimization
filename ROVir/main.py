import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import h5py
from methods import *
from methods import ROVIR_im
import math

dirs = [os.getcwd(), "data", "CalVolumeChannelCompression.h5"]
data_path = os.path.join(*dirs)

A_W = slice(22, 41)
B1_W = slice(42, -1)
B2_W = slice(0, 20)
A_H = slice(30, 45)
B1_H = slice(20, 57)

DEBUGGING = False

if __name__ == '__main__':
    img_data = extract_hf5_data(data_path)

    regions = [A_W, A_H, B1_W, B1_H, B2_W]

    new_img = ROVir_im(img_data, regions)
    #fig, axs = plt.subplots(1, 2)

    #axs[0].imshow(img_data[:, :, 0], cmap='gray')
    #axs[1].imshow(img_data[:, :, 0],  cmap='gray')
    # plt.show()
