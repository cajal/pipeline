from scipy.interpolate import interp1d, interp2d
import numpy as np


def correct_motion(img, xymotion):
    assert isinstance(img, np.ndarray) and len(xymotion) == 2, 'Cannot correct stacks. Only 2D images please.'
    sz = img.shape
    y1, x1 = np.ogrid[0: sz[0], 0: sz[1]]
    y2, x2 = [np.arange(sz[0]) + xymotion[1], np.arange(sz[1]) + xymotion[0]]

    interp = interp2d(x1, y1, img, kind='cubic')
    img = interp(x2, y2)

    return img


def correct_raster(img, raster_phase, fill_fraction ):
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

def plot_raster():

    import matplotlib.pyplot as plt
    import seaborn as sns
    from pipeline import preprocess
    from tiffreader import TIFFReader
    reader = TIFFReader('/Users/titan/data/cache/11676_4_00006_00009.tif')
    img=reader[:,:,0,0,100]
    raster_phase = (preprocess.Prepare.Galvo() & dict(animal_id=11676, session=4, scan_idx=6)).fetch1['raster_phase']
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
