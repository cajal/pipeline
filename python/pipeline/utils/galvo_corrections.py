from pipeline import PipelineException
from scipy.interpolate import interp1d, RectBivariateSpline, interp2d
import numpy as np

def correct_motion(scan, xy_motion, in_place=True):
    """ Motion correction for two-photon scans.

    Moves each image in the scan x number of pixels to the left, y number of pixels up and
    resamples. Works for 2-d arrays and up. Only for square images (neither interp2d nor
    RectBivariateSpline can deal with non-rectangular grids).

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
    :param list/np.aray xy_motion: Volume with x, y motion offsets for each image in the
    first two dimensions
    :param bool in_place: If True (default), the original array is modified in memory.
    :return: Motion corrected scan
    :rtype: Same as scan if scan.dtype is subtype of np.float, else np.double.
    :raises: PipelineException
    """
    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException('Scan needs to be a numpy array.')
    if scan.ndim < 2:
        raise PipelineException('Scan with less than 2 dimensions.')
    if not np.issubdtype(scan.dtype, np.float):
        print('Changing scan type from', str(scan.dtype), 'to double')
        scan = np.double(scan)

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]

    # If copy is needed
    if not in_place:
        scan = scan.copy()

    # Reshape input (to deal with more than 2-D volumes)
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    num_images = reshaped_scan.shape[-1]
    reshaped_xy = np.reshape(xy_motion, (2, -1))

    # Over every image (as long as the x and y offset is defined)
    for i in range(num_images):
        [x_offset, y_offset] = reshaped_xy[:, i]
        if not np.isnan(x_offset) and not np.isnan(y_offset):

            # Get current image
            image = reshaped_scan[:, :, i]

            # Create interpolation function
            interp_function = interp2d(range(image_width), range(image_height), image,
                                      kind= 'cubic')

            # Evaluate on the original image plus offsets
            reshaped_scan[:, :, i] = interp_function(np.arange(image_width) + x_offset,
                                                     np.arange(image_height) + y_offset)

    scan = np.reshape(reshaped_scan, original_shape)
    return scan


def correct_raster(scan, raster_phase, fill_fraction, in_place=True):
    """ Raster correction for resonant scanners.

    Corrects two-photon images in n-dimensional scans, usual shape is [image_height,
    image_width, channels, slices, num_frames]. Works for 2-d arrays and up (assuming
    images are in the first two dimensions). Based on Matlab implementation of
    ne7.ip.correctRaster()

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
    :param float raster_phase: Phase difference beetween odd and even lines.
    :param float fill_fraction: Ratio between active acquisition and total length of the
    scan line.
    :param bool in_place: If True (default), the original array is modified in memory.
    :return: Raster-corrected scan.
    :rtype: Same as scan if scan.dtype is subtype of np.float, else np.double.
    :raises: PipelineException
    """
    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException('Scan needs to be a numpy array.')
    if scan.ndim < 2:
        raise PipelineException('Scan with less than 2 dimensions.')
    if not np.issubdtype(scan.dtype, np.float):
        print('Changing scan type from', str(scan.dtype), 'to double')
        scan = np.double(scan)

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]
    half_width = image_width / 2

    # If copy is needed
    if not in_place:
        scan = scan.copy()

    # ??
    index  = np.linspace(-half_width + 0.5, half_width - 0.5, image_width) / half_width
    time_index = np.arcsin(index * fill_fraction)
    interp_points_even = np.sin(time_index + raster_phase) / fill_fraction
    interp_points_odd = np.sin(time_index - raster_phase) / fill_fraction

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