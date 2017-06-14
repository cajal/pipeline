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


def mirrconv(s, h):
    """
    Convolution where the ends are mirrored to have the same signal statistic.

    Only works for one dimensional arrays/

    :param s: signal
    :param h: filter (length must be odd)
    :return: filtered signal of the same length
    """
    assert s.ndim == 1, "Only one dimensional signals allowed!"
    assert h.ndim == 1, "Only one dimensional filters allowed!"
    assert len(h) % 2 == 1, "Filter must have odd length"

    n = h.size // 2
    return np.convolve(np.hstack((s[n-1::-1], s, s[:-n-1:-1])), h, mode='valid')