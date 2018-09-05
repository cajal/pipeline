import numpy as np
from sklearn.linear_model import TheilSenRegressor
from scipy import signal


def compute_quantal_size(scan):
    """ Estimate the unit change in calcium response corresponding to a unit change in
    pixel intensity (dubbed quantal size, lower is better).

    Assumes images are stationary from one timestep to the next. Uses it to calculate a
    measure of noise per bright intensity (which increases linearly given that imaging
    noise is poisson), fits a line to it and uses the slope as the estimate.

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).

    :returns: int minimum pixel value in the scan (that appears a min number of times)
    :returns: int maximum pixel value in the scan (that appears a min number of times)
    :returns: np.array pixel intensities used for the estimation.
    :returns: np.array noise variances used for the estimation.
    :returns: float the estimated quantal size
    :returns: float the estimated zero value
    """
    # Set some params
    num_frames = scan.shape[2]
    min_count = num_frames * 0.1  # pixel values with fewer appearances will be ignored
    max_acceptable_intensity = 3000  # pixel values higher than this will be ignored

    # Make sure field is at least 32 bytes (int16 overflows if summed to itself)
    scan = scan.astype(np.float32, copy=False)

    # Create pixel values at each position in field
    eps = 1e-4 # needed for np.round to not be biased towards even numbers (0.5 -> 1, 1.5 -> 2, 2.5 -> 3, etc.)
    pixels = np.round((scan[:, :, :-1] + scan[:, :, 1:]) / 2 + eps)
    pixels = pixels.astype(np.int16 if np.max(abs(pixels)) < 2 ** 15 else np.int32)

    # Compute a good range of pixel values (common, not too bright values)
    unique_pixels, counts = np.unique(pixels, return_counts=True)
    min_intensity = min(unique_pixels[counts > min_count])
    max_intensity = max(unique_pixels[counts > min_count])
    max_acceptable_intensity = min(max_intensity, max_acceptable_intensity)
    pixels_mask = np.logical_and(pixels >= min_intensity, pixels <= max_acceptable_intensity)

    # Select pixels in good range
    pixels = pixels[pixels_mask]
    unique_pixels, counts = np.unique(pixels, return_counts=True)

    # Compute noise variance
    variances = ((scan[:, :, :-1] - scan[:, :, 1:]) ** 2 / 2)[pixels_mask]
    pixels -= min_intensity
    variance_sum = np.zeros(len(unique_pixels)) # sum of variances per pixel value
    for i in range(0, len(pixels), int(1e8)):  # chunk it for memory efficiency
        variance_sum += np.bincount(pixels[i: i + int(1e8)], weights=variances[i: i + int(1e8)],
                                    minlength=np.ptp(unique_pixels) + 1)[unique_pixels - min_intensity]
    unique_variances = variance_sum / counts # average variance per intensity

    # Compute quantal size (by fitting a linear regressor to predict the variance from intensity)
    X = unique_pixels.reshape(-1, 1)
    y = unique_variances
    model = TheilSenRegressor() # robust regression
    model.fit(X, y)
    quantal_size = model.coef_[0]
    zero_level = - model.intercept_ / model.coef_[0]

    return (min_intensity, max_intensity, unique_pixels, unique_variances,
           quantal_size, zero_level)


def find_peaks(trace):
    """ Find local peaks in the signal and compute prominence and width at half
    prominence. Similar to Matlab's findpeaks.

    :param np.array trace: 1-d signal vector.

    :returns: np.array with indices for each peak.
    :returns: list with prominences per peak.
    :returns: list with width per peak.
    """
    # Get peaks (local maxima)
    peak_indices = signal.argrelmax(trace)[0]

    # Compute prominence and width per peak
    prominences = []
    widths = []
    for index in peak_indices:
        # Find the level of the highest valley encircling the peak
        for left in range(index - 1, -1, -1):
            if trace[left] > trace[index]:
                break
        for right in range(index + 1, len(trace)):
            if trace[right] > trace[index]:
                break
        contour_level = max(min(trace[left: index]), min(trace[index + 1: right + 1]))

        # Compute prominence
        prominence = trace[index] - contour_level
        prominences.append(prominence)

        # Find left and right indices at half prominence
        half_prominence = trace[index] - prominence / 2
        for k in range(index - 1, -1, -1):
            if trace[k] <= half_prominence:
                left = k + (half_prominence - trace[k]) / (trace[k + 1] - trace[k])
                break
        for k in range(index + 1, len(trace)):
            if trace[k] <= half_prominence:
                right = k - 1 + (half_prominence - trace[k - 1]) / (trace[k] - trace[k - 1])
                break

        # Compute width
        width = right - left
        widths.append(width)

    return peak_indices, prominences, widths