import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from methods import *

import time

dirs = [os.getcwd(), "data", "Slice44-AllChannels.nii"]
data_path = os.path.join(*dirs)

A_W = slice(110, 410)
B1_W = slice(311, -1)
B2_W = slice(0, 100)
A_H = slice(240, 405)
B1_H = slice(100, 450)

DEBUGGING = True

if __name__ == '__main__':
    print(data_path)
    # Testing to reconstruct a .nii image
    img = nib.load(data_path)

    height = img.shape[0]
    width = img.shape[1]
    num_coils = img.shape[2]
    #lowf = int(input("lowf = "))
    lowf = 8

    print(img.shape)

    img_np = np.array(img.dataobj)

    prev_img = combine_images(img_np)
    fig, axs = plt.subplots(1, 2)
    im1 = axs[0].imshow(prev_img, cmap='gray')
    axs[0].set_title('Before ROVir')
    regions = [A_W, A_H, B1_W, B1_H, B2_W]

    rovir_coils = ROVir(img_np, regions,  lowf)

    if not DEBUGGING:
        new_img = combine_images(rovir_coils)
        im2 = axs[1].imshow(new_img, cmap='gray')
        axs[1].set_title('After ROVir')
        for im in plt.gca().get_images():
            im.set_clim(0, 10000)
        im1.set_clim(0, 1000)
        plt.show()
    else:
        pass
