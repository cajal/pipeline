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


def compute_correlation_image_2(scan):
    """ Compute the correlation image for the given scan.

    A dynamic programming implementation of compute_correlation_image. Overhead of
    bookkeeping seems to offset the gains of less computation.

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).

    :returns: Correlation image. 2-dimensional array (image_height x image_width).
    :rtype np.array
    """
    # Get image dimensions
    image_height, image_width, num_frames = scan.shape

    # Calculate mean and standard deviation per pixel
    mean_image = np.mean(scan, axis=-1)
    # std_image = np.std(scan, axis=-1) creates a copy of scan, the loop below does not
    std_image = np.empty_like(mean_image)
    for y, x in product(range(image_height), range(image_width)):
        std_image[y, x] = np.sqrt(np.mean((scan[y, x, :] - mean_image[y, x]) ** 2))

    # # Create repository for correlations (dictionary of dictionaries, (y,x) as indices)
    # correlations = {key: {} for key in product(range(image_height), range(image_width))}
    #
    # # Calculate pixel correlations to each of its 8 neighbors
    # for y, x in product(range(image_height), range(image_width)):
    #     pixel_deviations = scan[y, x] - mean_image[y, x]
    #
    #     yrange = range(max(y - 1, 0), min(y + 2, image_height))
    #     xrange = range(max(x - 1, 0), min(x + 2, image_width))
    #     for y2, x2 in product(yrange, xrange):
    #         if (y2, x2) != (y, x) and (y2, x2) not in correlations[(y, x)]:
    #             # Calculate correlation between pixels
    #             pixel2_deviations = scan[y2, x2] - mean_image[y2, x2]
    #             numerator = np.mean(pixel_deviations * pixel2_deviations)
    #             rho = numerator / (std_image[y, x] * std_image[y2, x2])
    #
    #             # Add it to the dictionaries
    #             correlations[(y, x)][(y2, x2)] = rho
    #             correlations[(y2, x2)][(y, x)] = rho
    #
    # # Create correlation image
    # correlation_image = np.zeros_like(mean_image)
    # for y, x in product(range(image_height), range(image_width)):
    #     yrange = range(max(y - 1, 0), min(y + 2, image_height))
    #     xrange = range(max(x - 1, 0), min(x + 2, image_width))
    #     for y2, x2 in product(yrange, xrange):
    #         if (y2, x2) != (y, x):
    #             correlation_image[y, x] += correlations[(y, x)][(y2, x2)]


    # Create repository for correlations (list of dictionaries)
    correlations = [[{} for j in range(image_width)] for i in range(image_height)]

    # Calculate pixel correlations to each of its 8 neighbors
    for y, x in product(range(image_height), range(image_width)):
        pixel_deviations = scan[y, x] - mean_image[y, x]

        yrange = range(max(y - 1, 0), min(y + 2, image_height))
        xrange = range(max(x - 1, 0), min(x + 2, image_width))
        for y2, x2 in product(yrange, xrange):
            if (y2, x2) != (y, x) and (y2, x2) not in correlations[y][x]:
                # Calculate correlation between pixels
                pixel2_deviations = scan[y2, x2] - mean_image[y2, x2]
                numerator = np.mean(pixel_deviations * pixel2_deviations)
                rho = numerator / (std_image[y, x] * std_image[y2, x2])

                # Add it to the dictionaries
                correlations[y][x][(y2, x2)] = rho
                correlations[y2][x2][(y, x)] = rho

    # Create correlation image
    correlation_image = np.zeros_like(mean_image)
    for y, x in product(range(image_height), range(image_width)):
        yrange = range(max(y - 1, 0), min(y + 2, image_height))
        xrange = range(max(x - 1, 0), min(x + 2, image_width))
        for y2, x2 in product(yrange, xrange):
            if (y2, x2) != (y, x):
                correlation_image[y, x] += correlations[y][x][(y2, x2)]

    # Average each pixel by the number of correlations it summed over
    sums = np.full_like(mean_image, 5)  # 5 in borders
    sums[1:-1, 1:-1] = 8  # 8 in the center
    sums[[0, 0, -1, -1], [0, -1, 0, -1]] = 3  # 3 in corners
    correlation_image = correlation_image / sums

    return correlation_image