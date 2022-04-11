import numpy as np
import matplotlib.pyplot as plt


def combine_images(channels):
    # Start by combining images by the root sum of squares

    reconstructed_image = np.zeros(channels.shape[:1])
    for i in range(channels.shape[2]):
        reconstructed_image = np.sqrt(np.square(channels[:, :, i])
                                      + np.square(reconstructed_image))

    return reconstructed_image


def intensity_plot(image, line):
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(image)
    axs[0].set_title('Image')

    axs[1].set_title('Intensity graph')
