import numpy as np


def notnan(x, start=0, increment=1):
    while np.isnan(x[start]) and 0 <= start < len(x):
        start += increment
    return start


def fill_nans(x):
    """
    :param x:  1D array  -- will
    :return: the array with nans interpolated
    The input argument is modified.
    """
    nans = np.isnan(x)
    x[nans] = 0 if nans.all() else np.interp(nans.nonzero()[0], (~nans).nonzero()[0], x[~nans])
    return x


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def mirrconv(signal, f):
    """ Convolution with mirrored ends to avoid edge artifacts.

    :param np.array signal: One-dimensional signal.
    :param np.array f: One-dimensional filter (length should be odd).
    :returns: Filtered signal (same length as signal)
    """
    if signal.ndim != 1 or f.ndim != 1:
       raise ValueError('Only one-dimensional signals allowed.')
    if len(f) % 2 != 1:
        raise ValueError('Filter must have odd length')
    if len(f) < 3:
        return signal

    n = len(f) // 2
    padded_signal = np.hstack((signal[n - 1::-1], signal, signal[:-n - 1:-1]))
    filtered_signal = np.convolve(padded_signal, f, mode='valid')

    return filtered_signal


def float2uint8(scan):
    """ Converts an scan (or image) from floats to uint8 (preserving the range)."""
    scan = (scan - scan.min()) / (scan.max() - scan.min())
    scan = (scan * 255).astype(np.uint8, copy=False)
    return scan


def spaced_max(x, min_interval):
    """ Find all local peaks that are at least min_interval indices apart."""
    from scipy.signal import argrelmax

    peaks = argrelmax(x)[0]
    if len(peaks) != 0:
        new_peaks = [peaks[0]]
        for next_candidate in peaks[1:]:
            if next_candidate - new_peaks[-1] >= min_interval:
                new_peaks.append(next_candidate)
            elif x[next_candidate] > x[new_peaks[-1]]:
                new_peaks[-1] = next_candidate
        peaks = np.array(new_peaks)

    return peaks