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