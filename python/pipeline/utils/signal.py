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


def local_max(x):
    N = x.size
    x = x.ravel()
    b1 = x[:N-1] <= x[1:] # left <= right
    b2 = x[:N-1] >  x[1:] # left > right
    k = np.where(b1[:-1] & b2[1:])[0] + 1
    if x[0]>x[1]:
        k = np.hstack((k, [0]))

    if x[-1]>x[-2]:
        k = np.hstack((k, [N-1]))
    k.sort()
    return k

def spaced_max(x, min_interval, thresh=None):
    peaks = local_max(x)
    if thresh is not None:
        peaks = peaks[x(peaks) > thresh]

    if len(peaks) == 0:
        idx = []
    else:
        idx = [peaks[0]]
        for i in peaks[1:]:
            if i - idx[-1] >= min_interval:
                idx.append(i)
            elif x[i] > x[idx[-1]]:
                idx[-1] = i
    return np.array(idx)

def longest_contiguous_block(idx):
    d = np.diff(idx)
    ix = np.hstack(([-1], np.where(d > 10*np.median(d))[0], [len(idx)]))
    f = [idx[ix[i] + 1: ix[i+1]] for i in range(len(ix)-1)]
    return f[np.argmax([len(e) for e in f])]

