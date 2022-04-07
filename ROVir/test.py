import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from methods import *

if __name__ == '__main__':

    data_path = os.path.join(
        os.getcwd(), "data", "CalVolumeChannelCompression.h5")

    f = h5py.File(data_path, 'r')
    print(type(f))
    datasets = list(f.keys())

    images = f[datasets[1]]

    print(images.shape)

    img_ = images[0, :, :, :]

    #img_ = img_.H.dot(img_)

    print(img_.shape)

    f = np.vectorize(convert_void_toC)

    img_ = f(img_)

    img_ = img_.sum(axis=0)

    print(img_.dtype)

    print(img_.shape)

    img_ = np.matrix(img_).T
    #img_ = img_.H.dot(img_)

    plt.imshow(img_.real, cmap='bone')
    plt.show()
