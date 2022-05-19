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
A_H = slice(220, 390)
B1_H = slice(80, 450)

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

    img_np = np.transpose(
        np.array(img.dataobj), 
        axes=(0,2,1)
    )
    #img_np.transpose(0,2,1)
    

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
    
    plot_image(
        prev_img,
        save=True
    )

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

    plot_intensities(
        prev_img, new_img, 
        267, save=True
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
