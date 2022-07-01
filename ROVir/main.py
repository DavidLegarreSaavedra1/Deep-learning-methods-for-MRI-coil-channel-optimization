from enum import auto
from pyparsing import line
from methods import *
from icecream import ic
from ast import Slice
from PIL import Image
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import h5py
import cv2 as cv
import scipy.misc
import matplotlib.image as mpimg
import matplotlib
import seaborn as sns

# Matplotlib configuration
matplotlib.use('tkagg')
plt.style.use('ggplot')
plt.rcParams['image.cmap'] = "gray"
plt.rcParams['figure.dpi'] = "100"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = "False"
plt.rcParams["savefig.transparent"] = "True"

# Paths
dirs = [os.getcwd(), "data"]
data_path = os.path.join(*dirs)

# Constants
A_W = slice(200, 390)
B1_W = slice(0, 150)
B2_W = slice(430, -1)
A_H = slice(120, 280)
B1_H = slice(50, 450)

LINE = 300

regions = [A_W, A_H, B1_W, B1_H, B2_W]

# Heart area_color
H_W = slice(120, 310)
H_H = slice(230, 400)


def main():

    img = nib.load(os.path.join(data_path, "Slice44-AllChannels.nii"))

    height = img.shape[0]
    width = img.shape[1]
    num_coils = img.shape[2]
    # lowf = int(input("lowf = "))
    lowf = 8

    img_np = np.array(img.dataobj)
    img_np = np.flip(img_np, [0, 1])

    prev_img = combine_images(img_np)
    prev_img = auto_contrast(prev_img, 0.99)

    regions = [A_W, A_H, B1_W, B1_H, B2_W]

    rovir_coils, _ = ROVir(img_np, regions,  lowf)

    new_img = combine_images(rovir_coils)
    #bot_img = combine_images(bot_coils)

    nmax1 = np.max(prev_img[H_H, H_W])
    nmax2 = np.max(new_img[H_H, H_W])

    plot_images(
        prev_img,
        "Before ROVir",
        int(nmax1)*2.5,
        new_img,
        "After ROVir",
        nmax2=800,
        save=True,
        saveTitle="Comparing Rovir"
    )

    plot_intensities(
        prev_img, new_img,
        235, save=True
    )

    i = 0
    for coil in rovir_coils.reshape(-1, 512, 512):
        plt.imshow(coil)
        plt.show()
        coil = cv.normalize(
            coil, None, 0, 255, cv.NORM_MINMAX
        )
        cv.imwrite("virtual/coil"+str(i)+".png", coil)
        i += 1
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
