from ast import Slice
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import h5py
from pyparsing import line
from methods import *
from icecream import ic

dirs = [os.getcwd(), "data"]
data_path = os.path.join(*dirs)

A_W = slice(100, 350)
B1_W = slice(0, 90)
B2_W = slice(430, -1)
A_H = slice(220, 410)
B1_H = slice(90, 440)

LINE = 300
DEBUGGING = False

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
    lowf = 15

    img_np = np.array(img.dataobj)

    prev_img = combine_images(img_np)

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

    plot_images(prev_img,
                "Before ROVir",
                int(nmax1)*2.5,
                new_img,
                "After ROVir",
                int(nmax2)*2
                )

    plot_images(
        new_img,
        "Top coils",
        int(nmax2),
        bot_img,
        "Bottom coils",
        int(nmax1)*2.5
    )

    intensity_plot(prev_img, 270, 'Before ROVir')
    intensity_plot(new_img, 270, 'After ROVir')
    plt.show()


if __name__ == '__main__':
    main()
