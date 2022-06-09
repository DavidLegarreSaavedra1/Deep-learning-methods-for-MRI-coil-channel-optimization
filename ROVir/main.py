import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import h5py
from methods import *
from methods import ROVIR_im
import math
import matplotlib

matplotlib.use('tkagg')
plt.style.use('ggplot')
plt.rcParams['image.cmap'] = "gray"
plt.rcParams['figure.dpi'] = "100"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = "False"
plt.rcParams["savefig.transparent"] = "True"


dirs = [os.getcwd(), "data"]
data_path = os.path.join(*dirs)

A_W = slice(22, 41)
B1_W = slice(42, -1)
B2_W = slice(0, 20)
A_H = slice(30, 45)
B1_H = slice(20, 57)

regions = [A_W, A_H, B1_W, B1_H, B2_W]

DEBUGGING = False

if __name__ == '__main__':
    sens = extract_hf5_data(os.path.join(
        data_path, "CalVolumeChannelCompression.h5"))

    img_data = nib.load(os.path.join(data_path, "Slice44-AllChannels.nii"))

    img_np = np.array(img_data.dataobj)
    prev_img = combine_images(img_np)

    weights, topNv = ROVir_im(sens, regions, 8)
    print(f'{topNv=}')

    plot_coils(img_np)

    Vcoils = generate_virtual_coils(img_np, weights, topNv)
    fig, axs = plt.subplots(1, 2)

    new_img = combine_images(Vcoils)

    axs[0].imshow(prev_img, cmap='bone')
    axs[1].imshow(new_img,  cmap='gray')
    plt.show()
