from configparser import Interpolation
import numpy as np
import matplotlib.pyplot as plt


def combine_images(channels):
    # Start by combining images by the root sum of squares

    reconstructed_image = np.zeros(channels.shape[:1])
    for i in range(channels.shape[2]):
        reconstructed_image = np.sqrt(np.square(channels[:, :, i])
                                      + np.square(reconstructed_image))

    return reconstructed_image


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


def plot_mask_A(image, mask):
    fig, ax = plt.subplots()
    m_h, m_w = mask

    im = ax.imshow(image, cmap='gray')
    im.set_clim(0, 1500)

    ax.plot(
        [m_w[0], m_w[1]],
        [m_h[0], m_h[0]],
        color='blue'
    )
    ax.plot(
        [m_w[0], m_w[1]],
        [m_h[1], m_h[1]],
        color='blue'
    )
    ax.plot(
        [m_w[0], m_w[0]],
        [m_h[0], m_h[1]],
        color='blue'
    )
    ax.plot(
        [m_w[1], m_w[1]],
        [m_h[0], m_h[1]],
        color='blue'
    )


def plot_masks(image, maskA, maskB, title=''):
    plt.figure()
    im = plt.imshow(image, cmap='gray')
    im.set_clim(0, np.max(image)/2)
    plt.imshow(maskA, alpha=0.25)
    plt.imshow(maskB, alpha=0.25)
    plt.title(title)


def plot_images(img1, title1, nmax1, img2, title2, nmax2):
    fig, axs = plt.subplots(1, 2)
    im1 = axs[0].imshow(img1,
                        cmap='gray')
    axs[0].set_title(title1)
    im1.set_clim(0, nmax1)

    im2 = axs[1].imshow(img2,
                        cmap='gray')
    axs[1].set_title(title2)
    im2.set_clim(0, nmax2)


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
    fig.suptitle(title, fontsize=16)
    return
