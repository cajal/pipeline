import numpy as np
from scipy import ndimage


def lcn(image, sigmas=(12, 12)):
    """ Local contrast normalization.

    Normalize each pixel using mean and stddev computed on a local neighborhood.

    We use gaussian filters rather than uniform filters to compute the local mean and std
    to soften the effect of edges. Essentially we are using a fuzzy local neighborhood.
    Equivalent using a hard defintion of neighborhood will be:
        local_mean = ndimage.uniform_filter(image, size=(32, 32))

    :param np.array image: Array with raw two-photon images.
    :param tuple sigmas: List with sigmas (one per axis) to use for the gaussian filter.
        Smaller values result in more local neighborhoods. 15-30 microns should work fine
    """
    local_mean = ndimage.gaussian_filter(image, sigmas)
    local_var = ndimage.gaussian_filter(image ** 2, sigmas) - local_mean ** 2
    local_std = np.sqrt(np.clip(local_var, a_min=0, a_max=None))
    norm = (image - local_mean) / (local_std + 1e-7)

    return norm


def sharpen_2pimage(image, laplace_sigma=0.7, low_percentile=3, high_percentile=99.9):
    """ Apply a laplacian filter, clip pixel range and normalize.

    :param np.array image: Array with raw two-photon images.
    :param float laplace_sigma: Sigma of the gaussian used in the laplace filter.
    :param float low_percentile, high_percentile: Percentiles at which to clip.

    :returns: Array of same shape as input. Sharpened image.
    """
    sharpened = image - ndimage.gaussian_laplace(image, laplace_sigma)
    clipped = np.clip(sharpened, *np.percentile(sharpened, [low_percentile, high_percentile]))
    norm = (clipped - clipped.mean()) / (clipped.max() - clipped.min() + 1e-7)
    return norm


def create_correlation_image(scan):
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
    from itertools import product

     # Get image dimensions
    image_height, image_width, num_frames = scan.shape

    # Compute deviations from the mean (in place)
    mean_image = np.mean(scan, axis=-1, keepdims=True)
    scan -= mean_image # in place
    deviations = scan

    # Calculate (unnormalized) standard deviation per pixel
    stddev_image = np.empty([image_height, image_width])
    for y, x in product(range(image_height), range(image_width)):
        stddev_image[y, x] = np.sum(deviations[y, x] ** 2)
    stddev_image = np.sqrt(stddev_image)
    # we don't use np.sum(deviations**2, axis=-1) because it creates a copy of the scan

    # Cut a 3 x 3 square around each pixel and compute their (mean) pair-wise correlation.
    correlation_image = np.empty([image_height, image_width])
    for y, x in product(range(image_height), range(image_width)):
            yslice = slice(max(y - 1, 0), min(y + 2, image_height))
            xslice = slice(max(x - 1, 0), min(x + 2, image_width))

            numerator = np.inner(deviations[yslice, xslice], deviations[y, x])
            correlations = numerator / stddev_image[yslice, xslice]
            correlations[min(1, y), min(1, x)] = 0
            correlation_image[y, x] = np.sum(correlations) /  (correlations.size - 1)
    correlation_image /= stddev_image

    # Return scan back to original values
    scan += mean_image

    return correlation_image