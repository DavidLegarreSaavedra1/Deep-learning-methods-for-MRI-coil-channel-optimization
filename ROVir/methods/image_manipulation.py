from configparser import Interpolation
from .matrix_manipulation import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as image


def combine_images(channels):
    # Start by combining images by the root sum of squares

    reconstructed_image = np.zeros(channels.shape[:1])
    for i in range(channels.shape[2]):
        reconstructed_image = (np.square(channels[:, :, i])
                               + (reconstructed_image))

    return np.sqrt(reconstructed_image)


def auto_contrast(image, q=.99, dim=None):
    flat = image.flatten()
    sort_flat = np.sort(flat)
    nval = len(image[image < 50])
    limit = np.quantile(sort_flat[nval:], q)
    print(limit)
    flat[flat > limit] = limit
    flat[flat < 100] = 100
    if dim:
        return flat.reshape(*image.shape[:2], dim)
    return flat.reshape(*image.shape[:2])


def intensity_plot(image, height, Title):
    fig, axs = plt.subplots(1, 2)

    im = axs[0].imshow(image, cmap='gray')
    axs[0].plot(
        [0, image.shape[0]-1],
        [height, height],
        color='green'
    )
    axs[0].set_title(Title)

    axs[1].plot(
        np.linspace(0, image.shape[0]-1, image.shape[0]),
        image[height, :],
        color='gray'
    )

    #axs[1].set_title("Intensity plot")
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Intensity')

    im.set_clim(0, np.max(image)/2)


def plot_masks(image, maskA, maskB,
               title='', save=False):

    fig, axs = plt.subplots(1, 1)

    maska = np.empty((*maskA.shape, 3))
    maska[maskA > 0] = (0.0, 1.0, 0.0)

    maskb = np.empty((*maskB.shape, 3))
    maskb[maskB > 0] = (1.0, 0.0, 0.0)

    im = axs.imshow(image, cmap='gray')
    im.set_clim(0, np.max(image)/2)

    axs.imshow(maska, alpha=0.5)
    axs.imshow(maskb, alpha=0.5)
    axs.set_title(title)
    if save:
        plt.savefig("mask.png", transparent=True)


def save_image(img, title):
    im = image.fromarray(img)
    im.convert('L').save(title+".png")


def plot_images(img1, title1, nmax1, img2,
                title2, nmax2, save=False,
                saveTitle=""):
    fig, axs = plt.subplots(1, 2)
    im1 = axs[0].imshow(
        img1,cmap='gray',
        vmin=0, vmax=nmax1)
    axs[0].set_title(title1)

    im2 = axs[1].imshow(
        img2,cmap='gray',
        vmin=0, vmax=nmax2
    )
    axs[1].set_title(title2)

    if save:
        plt.savefig(saveTitle+".png", transparent=True)


def plot_coils(coils, title=''):
    ncoils = coils.shape[-1]
    x = int(np.floor(np.sqrt(ncoils)))
    fig, axs = plt.subplots(x, x)
    i = 0
    for ax in axs.reshape(ncoils):
        im = ax.imshow(coils[..., i], cmap='gray',
                       vmin=0, vmax=600)
        ax.set_title("coil: "+str(i))
        im.set_clim(0, 1)
        i += 1
    fig.suptitle(title, fontsize=12)
    plt.subplots_adjust(
        left=0.125,
        right=0.9,
        top=0.9,
        bottom=0.1,
        wspace=0.2,
        hspace=0.4
    )


def plot_intensities(img1, img2, height, save=False):
    #fig, axs = plt.subplots(1,2)

    ax0 = plt.subplot(221)
    ax0.imshow(
        img1, cmap='gray'
    )
    ax0.plot(
        [0, img1.shape[0]-1],
        [height, height],
        color='red'
    )
    ax0.set_title('Before ROVir')

    ax1 = plt.subplot(222)
    im2 = ax1.imshow(
        img2, cmap='gray',
        vmin=0, vmax=800
    )
    ax1.plot(
        [0, img2.shape[0]-1],
        [height, height],
        color='green'
    )
    ax1.set_title('After ROVir')

    ax2 = plt.subplot(212)
    tmp = moving_average(img1[height, :], 6)
    line1, = ax2.plot(
        np.linspace(0, img1.shape[0]-1, img1.shape[0]-5),
        tmp,
        color='red'
    )

    tmp = moving_average(img2[height, :], 6)
    line2, = ax2.plot(
        np.linspace(0, img2.shape[0]-1, img2.shape[0]-5),
        tmp,
        color='green'
    )

    ax2.set_xlabel("Position")
    ax2.set_ylabel("Intensity")
    ax2.legend([line1, line2], ["Before ROVir", "Afer ROVir"])
    if save:
        plt.savefig("intensities.png", transparent=True)


def plot_image(img1, title='', save=False):
    fig = plt.figure()
    ax = plt.subplot(111)
    nmax = np.max(img1)

    im = ax.imshow(img1, cmap='gray')
    ax.set_title(title)
    im.set_clim(0, nmax)
    if save:
        plt.savefig("image.png", transparent=True)
