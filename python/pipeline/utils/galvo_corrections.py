from .. import PipelineException
from scipy.interpolate import interp1d, RectBivariateSpline, interp2d
from scipy import signal
import numpy as np

def compute_motion_offsets(field, template):
    """ Compute offsets in x and y for rigid subpixel motion correction.
    
    :param np.array template: 2-d template image. Each frame in field is aligned to this.
    :param np.array field: 3-d field (image_height, image_width, num_frames). First two
        dimensions 
    
    :returns: Array with x, y motion offsets for each image in field ([2 x num_frames]).
        Positive offsets for right and down offsets, e.g., x_offset = 1 means image is 1 
        pixel to the right of template
    :rtype: np.array 
    """
    # Get some params
    num_frames = field.shape[2]
    max_y_offset = int(round(field.shape[0] * 0.10)) # max num_pixels images can be moved in y
    max_x_offset = int(round(field.shape[1] * 0.10))

    # If template and images are int, signal.correlate fails
    template = template.astype(np.float32)

    y_offsets = np.empty(num_frames)
    x_offsets = np.empty(num_frames)
    for i in range(num_frames):

        # Align pixels (convolving on template and finding highest value)
        center_patch = field[max_y_offset: -max_y_offset, max_x_offset: -max_x_offset, i]
        conv_template = signal.correlate(template, center_patch, mode='valid')
        y_offset, x_offset = np.unravel_index(np.argmax(conv_template), conv_template.shape)
        y_offsets[i] = y_offset - max_y_offset
        x_offsets[i] = x_offset - max_x_offset

        # TODO: Subpixel, check for black images, check too much motion

    return y_offsets, x_offsets


def compute_raster_phase(image):
    """"""
    print('Warning: Raster correction not implemented yet.')
    #TODO: Compute raster correction
    return 0

def correct_motion(scan, xy_motion, in_place=True):
    """ Motion correction for two-photon scans.

    Moves each image in the scan x number of pixels to the left, y number of pixels up and
    resamples. Works for 2-d arrays and up.

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
    :param list/np.array xy_motion: Volume with x, y motion offsets for each image in the
    first dimension: usually [2 x num_frames].
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

    if reshaped_xy.shape[-1] != reshaped_scan.shape[-1]:
        raise PipelineException('Scan and motion arrays have different dimensions')

    # Over every image (as long as the x and y offset is defined)
    for i in range(num_images):
        [x_offset, y_offset] = reshaped_xy[:, i]
        if not np.isnan(x_offset) and not np.isnan(y_offset):

            # Get current image
            image = reshaped_scan[:, :, i]

            # Create interpolation function
            interp_function = interp2d(range(image_width), range(image_height), image,
                                      kind= 'cubic', copy=False)

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
    :param float raster_phase: Phase difference between odd and even lines.
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

    # Create interpolation points for sinusoidal raster
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
