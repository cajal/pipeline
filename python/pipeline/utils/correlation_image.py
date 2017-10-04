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