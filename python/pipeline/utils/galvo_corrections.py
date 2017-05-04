from .. import PipelineException
from scipy.interpolate import interp1d, RectBivariateSpline, interp2d
from scipy import signal
import numpy as np

def compute_motion_offsets(field, template, align_subpixels=True):
    """ Compute offsets in x and y for rigid subpixel motion correction.
    
    A patch is taken from the center of each frame (the margin size that is not included
    represents the maximum possible pixel shifts that can be applied) and convolved 
    against the template to get the amount of shifting needed to align the center of the 
    frame with the template: 0 if center of frame = center of template. This is an 
    approximation of the full 2-d convolution but should be ok if center patch is big and 
    maximum shifts don't go over the assumed max_shift.
    
    A positive x_shift means that the image needs to be moved x_shift pixels left, a 
    positive y_shift means the image needs to be moved y_shift pixels up.
    
    :param np.array template: 2-d template image. Each frame in field is aligned to this.
    :param np.array field: 3-d field (image_height, image_width, num_frames). First two
        dimensions
    :param bool align_subpixels: Whether to apply subpixel alignment after aligning pixels
    
    :returns: Array with x, y motion offsets for each image in field ([2 x num_frames]).
    :rtype: np.array 
    """
    # Get some params
    image_height, image_width, num_frames = field.shape
    max_y_shift = int(round(image_height * 0.10))  # max num_pixels images can be moved in y
    max_x_shift = int(round(image_width * 0.10))

    # Assert template is float. If template and images are int, signal.correlate fails
    if not np.issubdtype(template.dtype, np.float):
        template = template.astype(np.float32)

    # Cut center patch of frames
    frame_centers = field[max_y_shift: -max_y_shift, max_x_shift: -max_x_shift, :]

    y_shifts = np.empty(num_frames)
    x_shifts = np.empty(num_frames)
    for i in range(num_frames):

        # Align pixels (convolving on template and finding highest value)
        conv_image = signal.correlate(template, frame_centers[:, :, i], mode='valid')
        y_shift, x_shift = np.unravel_index(np.argmax(conv_image), conv_image.shape)
        y_shifts[i] = max_y_shift - y_shift
        x_shifts[i] = max_x_shift - x_shift

        if align_subpixels:
            #TODO:
            pass

    # Smooth them
    #hanning = signal.hann(5)
    #y_shifts_smoothed = signal.convolve(y_shifts, hanning, mode='same')
    #x_shifts_smoothed = signal.convolve(x_shifts, hanning, mode='same')

    return y_shifts, x_shifts


def compute_raster_phase(image):
    """"""
    print('Warning: Raster correction not implemented yet.')
    #TODO: Compute raster correction
    return 0

def correct_motion(scan, xy_shifts, in_place=True):
    """ Motion correction for two-photon scans.

    Shifts each image in the scan x_shift pixels to the left and y_shift pixels up. Works 
    for 2-d arrays and up.

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
    :param list/np.array xy_shifts: Volume with x, y motion shifts for each image in the
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
    reshaped_xy = np.reshape(xy_shifts, (2, -1))

    if reshaped_xy.shape[-1] != reshaped_scan.shape[-1]:
        raise PipelineException('Scan and motion arrays have different dimensions')

    # Over every image (as long as the x and y shift is defined)
    for i in range(num_images):
        [x_shift, y_shift] = reshaped_xy[:, i]
        if not np.isnan(x_shift) and not np.isnan(y_shift):

            # Get current image
            image = reshaped_scan[:, :, i]

            # Create interpolation function
            interp_function = interp2d(range(image_width), range(image_height), image,
                                      kind='cubic', copy=False)

            # Evaluate on the original image plus offsets
            reshaped_scan[:, :, i] = interp_function(np.arange(image_width) + x_shift,
                                                     np.arange(image_height) + y_shift)

    scan = np.reshape(reshaped_scan, original_shape)
    return scan


def correct_raster(scan, raster_phase, fill_fraction, in_place=True):
    """ Raster correction for resonant scanners.

    Corrects two-photon images in n-dimensional scans, usual shape is [image_height,
    image_width, channels, slices, num_frames]. Works for 2-d arrays and up (assuming
    images are in the first two dimensions). Based on Matlab implementation of
    ne7.ip.correctRaster()
    
    Positive raster phase shifts even lines to the left and odd lines to the right. 
    Negative raster phase shifts even lines to the right and odd lines to the left.  

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
