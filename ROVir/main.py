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

# Matplotlib configuration
matplotlib.use('tkagg')
plt.style.use('ggplot')
plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['figure.dpi'] = "100"
plt.rcParams["savefig.bbox"] = "tight"

# Paths
dirs = [os.getcwd(), "data"]
data_path = os.path.join(*dirs)

# Constants
A_W = slice(200, 390)
B1_W = slice(0, 150)
B2_W = slice(430, -1)
A_H = slice(120, 280)
B1_H = slice(80, 450)

LINE = 300
DEBUGGING = False

regions = [A_W, A_H, B1_W, B1_H, B2_W]

# Heart area_color
H_W = slice(120, 310)
H_H = slice(230, 400)


def main():

    img = nib.load(os.path.join(data_path, "Slice44-AllChannels.nii"))
    print(type(img))

    height = img.shape[0]
    width = img.shape[1]
    num_coils = img.shape[2]
    # lowf = int(input("lowf = "))
    lowf = 10

    img_np = np.array(img.dataobj) 
    img_np = np.flip(img_np, [0,1])
    img_np = auto_contrast(img_np, 0.99, 25)


    prev_img = combine_images(img_np)
    prev_img = auto_contrast(prev_img, 0.99)

    print(prev_img.shape)
    prev_img_ = cv.normalize(prev_img, None, alpha=0,beta=255, norm_type=cv.NORM_MINMAX)
    print(prev_img.shape)


    cv.imwrite('prev_img.png', prev_img)

    fig, axs = plt.subplots(1, 2)
    regions = [A_W, A_H, B1_W, B1_H, B2_W]

    rovir_coils, bot_coils = ROVir(img_np, regions,  lowf)

    new_img = combine_images(rovir_coils)
    bot_img = combine_images(bot_coils)

    nmax1 = np.max(prev_img[H_H, H_W])
    nmax2 = np.max(new_img[H_H, H_W])

    save_image(
        prev_img,
        "prev_img"
    )
    
   # plot_image(
   #     prev_img,
   #     save=True
   # )

    plot_images(prev_img,
                "Before ROVir",
                int(nmax1)*2.5,
                new_img,
                "After ROVir",
                int(nmax2)*2,
                save=True
                )

    plot_images(
        new_img,
        "Top coils",
        int(nmax2),
        bot_img,
        "Bottom coils",
        int(nmax1)*2.5
    )

    print(prev_img.shape)
    print(new_img.shape)
    
    plot_intensities(
        prev_img, new_img, 
        235, save=True
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
