import numpy as np
from itertools import product

def compute_correlation_image(scan):
    """ Compute the correlation image for the given scan.

    At each pixel, we compute the correlation (over time) with each of its eight
    neighboring pixels and average them.

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).

    :returns: Correlation image. 2-dimensional array (image_height x image_width).
    :rtype np.array

    ..note:: Even though this code does not reuse the correlations between pixels for the
    next iteration it is as efficient in time and (slightly better in) memory than the
    dynamic programming implementation below. It may be due to vectorization usage.
    """
     # Get image dimensions
    image_height, image_width, num_frames = scan.shape

    # Calculate mean and sum of squared deviations per pixel
    mean_image = np.mean(scan, axis=-1, keepdims=True)
    sqdev_image = np.empty([image_height, image_width])
    for y, x in product(range(image_height), range(image_width)):
        sqdev_image[y, x] = np.sum((scan[y, x, :] - mean_image[y, x]) ** 2)
    # we don't use np.std because it creates a copy of scan, the loop does not.

    # Cut a 3 x 3 square around each pixel and compute their (mean) pair-wise correlation.
    correlation_image = np.empty([image_height, image_width])
    for y, x in product(range(image_height), range(image_width)):
            yslice = slice(max(y - 1, 0), min(y + 2, image_height))
            xslice = slice(max(x - 1, 0), min(x + 2, image_width))

            rho = _pixel_correlation(scan[yslice, xslice, :], mean_image[yslice, xslice],
                                     sqdev_image[yslice, xslice], min(1, y), min(1, x))
            correlation_image[y, x] = rho

    return correlation_image


def _pixel_correlation(patch, mean_patch, sqdev_patch, y, x):
    """ Computes the (average) pairwise correlation between the pixel at y, x and the rest
    of the pixels.

    :param np.array patch: m x n x num_frames, small patch from scan.
    :param np.array mean_patch: m x n, mean pixel value (over time).
    :param np.array sqdev_patch: m x n, standard deviation per pixel (over time).
    :param int y: Position of the pixel of interest in y.
    :param int x: Position of the pixel of interest in x.
    """
    deviations = patch - mean_patch
    numerator = np.sum(deviations * deviations[y, x, :], axis=-1)
    correlations = numerator / np.sqrt(sqdev_patch * sqdev_patch[y, x])
    correlations[y, x] = 0
    return np.sum(correlations) / (correlations.size - 1)