import numpy as np


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
    return np.convolve(np.stack((s[n-1::-1], s, s[:-n-1:-1])), h, mode='valid')