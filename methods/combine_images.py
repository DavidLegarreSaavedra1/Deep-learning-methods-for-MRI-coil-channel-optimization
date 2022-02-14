import numpy as np


def combine_images(channels):
    # Start by combining images by the root sum of squares

    reconstructed_image = np.zeros(channels.shape[:1])
    for i in range(channels.shape[2]):
        reconstructed_image = np.sqrt(np.square(channels[:, :, i])
                                      + np.square(reconstructed_image))

    return reconstructed_image
