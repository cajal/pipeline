#TODO: Relative import 'from .. import PipelineException'
from pipeline import PipelineException
from scipy.interpolate import interp1d, interp2d
import numpy as np

def correct_motion(img, xymotion):
    """
    motion correction for 2P scans.
    :param img: 2D image [x, y]
    :param xymotion: x, y motion offsets
    :return: motion corrected image [x, y]
    """
    assert isinstance(img, np.ndarray) and len(
        xymotion) == 2, 'Cannot correct stacks. Only 2D images please.'
    sz = img.shape
    y1, x1 = np.ogrid[0: sz[0], 0: sz[1]]
    y2, x2 = [np.arange(sz[0]) + xymotion[1], np.arange(sz[1]) + xymotion[0]]

    interp = interp2d(x1, y1, img, kind='cubic')
    img = interp(x2, y2)

    return img


def correct_raster(scan, raster_phase, fill_fraction, in_place=True):
    """ Raster correction for resonant scanners.

    Corrects EM images in n-dimensional scans, usual shape is [image_height, image_width,
    channels, slices, num_frames]. Woks for 2-d arrays and up (assuming images are in the
    first two dimensions). Based on Matlab implementation of ne7.ip.correctRaster()

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
    :param float raster_phase: Phase difference beetween odd and even lines.
    :param float fill_fraction: Ratio between active acquisition and total length of the
    scan line.
    :param bool in_place: If True (default), the original array is modified in memory.
    :return: Raster-corrected scan.
    :rtype: Same as scan.
    :raises: PipelineException: Scan is not an np.array with at least 2 dimensions.
    """
    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException('Scan needs to be a numpy array.')
    if scan.ndim < 2:
        raise PipelineException('Scan with less than 2 dimensions.')
    if not np.issubdtype(scan.dtype, np.float):
        print('Changing scan from', str(scan.dtype), 'to double')
        scan = np.double(scan)

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]
    half_width = image_width / 2

    # ??
    index  = np.linspace(-half_width + 0.5, half_width - 0.5, image_width) / half_width
    time_index = np.arcsin(index * fill_fraction)
    interp_points_even = np.sin(time_index + raster_phase) / fill_fraction
    interp_points_odd = np.sin(time_index - raster_phase) / fill_fraction

    # If copy is needed
    if not in_place:
        scan = scan.copy()

    # We iterate over every image in the scan (first 2 dimensions). Same correction
    # regardless of what channel, slice or frame they belong to.
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    num_images = reshaped_scan.shape[-1]
    for i in range(num_images):
        # Get current image
        image = reshaped_scan[:, :, i]
        mean_intensity = np.mean(image)  # to fill outside the interpolation range

        # Correct even rows of the image (0, 2, ...)
        interp_function = interp1d(index, image[::2, :], copy=False, bounds_error=False,
                                   fill_value=mean_intensity)
        reshaped_scan[::2, :, i] = interp_function(interp_points_even)

        # Correct odd rows of the image (1, 3, ...)
        interp_function = interp1d(index, image[1::2, :], copy=False, bounds_error=False,
                                   fill_value=mean_intensity)
        reshaped_scan[1::2, :, i] = interp_function(interp_points_odd)

    scan = np.reshape(reshaped_scan, original_shape)
    return scan


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
    img = reader[:, :, 0, 0, 100]
    raster_phase = (preprocess.Prepare.Galvo() & key).fetch1['raster_phase']
    newim = correct_raster(img, raster_phase, reader.fill_fraction)
    nnewim = correct_raster(newim, -raster_phase, reader.fill_fraction)
    print(np.mean(img - nnewim))

    plt.close()
    with sns.axes_style('white'):
        fig = plt.figure(figsize=(15, 8))
        gs = plt.GridSpec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img[:, :, 0, 0, 0], cmap=plt.cm.gray)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(newim[:, :, 0, 0, 0], cmap=plt.cm.gray)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(nnewim[:, :, 0, 0, 0], cmap=plt.cm.gray)
    plt.show()
