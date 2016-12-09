from scipy.interpolate import interp1d, interp2d
import numpy as np


def correct_motion(img, xymotion):
    """
    motion correction for 2P scans.
    :param img: 2D image [x, y]
    :param xymotion: x, y motion offsets
    :return: motion corrected image [x, y]
    """
    assert isinstance(img, np.ndarray) and len(xymotion) == 2, 'Cannot correct stacks. Only 2D images please.'
    sz = img.shape
    y1, x1 = np.ogrid[0: sz[0], 0: sz[1]]
    y2, x2 = [np.arange(sz[0]) + xymotion[1], np.arange(sz[1]) + xymotion[0]]

    interp = interp2d(x1, y1, img, kind='cubic')
    img = interp(x2, y2)

    return img


def correct_raster(img, raster_phase, fill_fraction):
    """
    raster correction for resonant scanners.
    :param img: 5D image [x, y, nchannel, nslice, nframe].
    :param raster_phase: phase difference beetween odd and even lines.
    :param fill_fraction: ratio between active acquisition and total length of the scan line. see scanimage.
    :return: raster-corrected image [x, y, nchannel, nslice, nframe].
    """
    img = np.array(img)
    assert img.ndim <= 5, 'Image size greater than 5D.'
    ix = np.arange(-img.shape[1]/2 + 0.5, img.shape[1]/2 + 0.5) / (img.shape[1]/2)

    tx = np.arcsin(ix * fill_fraction)
    for ichannel in range(img.shape[2]):
        for islice in range(img.shape[3]):
            for iframe in range(img.shape[4]):
                im = img[:, :, ichannel, islice, iframe].copy()
                extrapval = np.mean(im)
                img[::2, :, ichannel, islice, iframe] = interp1d(ix, im[::2, :], kind='linear', bounds_error=False,
                                                                  fill_value=extrapval)(np.sin(tx +
                                                                                        raster_phase)/fill_fraction)

                img[1::2, :, ichannel, islice, iframe] = interp1d(ix, im[1::2, :], kind='linear', bounds_error=-False,
                                                                  fill_value=extrapval)(np.sin(tx -
                                                                                        raster_phase)/fill_fraction)
    return img


def plot_raster(filename, key):
    """
    plot origin frame, raster-corrected frame, and reversed raster-corrected frame.
    :param filename:  full file path for tiff file.
    :param key: scan key for the tiff file.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pipeline import preprocess, experiment
    from tiffreader import TIFFReader
    reader = TIFFReader(filename)
    img=reader[:, :, 0, 0, 100]
    raster_phase = (preprocess.Prepare.Galvo() & key).fetch1['raster_phase']
    newim = correct_raster(img, raster_phase, reader.fill_fraction)
    nnewim = correct_raster(newim, -raster_phase, reader.fill_fraction)
    print(np.mean(img - nnewim))

    plt.close()
    with sns.axes_style('white'):
        fig=plt.figure(figsize=(15,8))
        gs=plt.GridSpec(1,3)
        ax1=fig.add_subplot(gs[0,0])
        ax1.imshow(img[:,:,0,0,0], cmap=plt.cm.gray)
        ax2=fig.add_subplot(gs[0,1])
        ax2.imshow(newim[:,:,0,0,0],cmap=plt.cm.gray)
        ax3=fig.add_subplot(gs[0,2])
        ax3.imshow(nnewim[:,:,0,0,0], cmap=plt.cm.gray)
    plt.show()
